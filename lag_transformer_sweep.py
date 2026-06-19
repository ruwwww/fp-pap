#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"
EXOG_COLS = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def infer_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() == DATE_COL.lower():
            return c
    return df.columns[0]


def cyclical(values: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * values.astype(float) / period
    return np.sin(theta), np.cos(theta)


def engineer_exog(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c != date_col:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    dt = pd.to_datetime(out[date_col])
    out["dow"] = dt.dt.dayofweek.astype(float)
    out["month"] = dt.dt.month.astype(float)
    out["doy"] = dt.dt.dayofyear.astype(float)
    s, c = cyclical(out["dow"], 7)
    out["dow_sin"] = s
    out["dow_cos"] = c
    s, c = cyclical(out["month"], 12)
    out["month_sin"] = s
    out["month_cos"] = c
    out["time_idx"] = np.arange(len(out), dtype=float)
    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["US_rate"] - out["BI_rate"]
    for c in EXOG_COLS + ["rate_spread"]:
        if c in out.columns:
            out[f"{c}_lag1"] = out[c].shift(1)
            out[f"{c}_lag5"] = out[c].shift(5)
            out[f"{c}_lag21"] = out[c].shift(21)
            out[f"{c}_diff1"] = out[c].diff()
    return out


def target_context(levels_history: Sequence[float]) -> dict[str, float]:
    h = pd.Series(levels_history, dtype=float)
    feats = {
        "usdidr_lag1": h.iloc[-1] if len(h) >= 1 else np.nan,
        "usdidr_lag5": h.iloc[-5] if len(h) >= 5 else np.nan,
        "usdidr_lag21": h.iloc[-21] if len(h) >= 21 else np.nan,
        "usdidr_lag63": h.iloc[-63] if len(h) >= 63 else np.nan,
        "usdidr_diff1": h.diff().iloc[-1] if len(h) >= 2 else np.nan,
    }
    return feats


def build_supervised(train_df: pd.DataFrame, exog_df: pd.DataFrame, target_mode: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    levels = pd.to_numeric(train_df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    rows = []
    ys = []
    for t in range(64, len(train_df)):
        row = {c: exog_df.iloc[t][c] for c in exog_df.columns if c != DATE_COL}
        row.update(target_context(levels[:t]))
        rows.append(row)
        if target_mode == "return":
            ys.append(float(np.log(levels[t] / levels[t - 1])))
        else:
            ys.append(float(levels[t] - levels[t - 1]))
    X = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    y = pd.Series(ys, dtype=float)
    valid = X.notna().all(axis=1) & y.notna()
    dates = pd.to_datetime(train_df[DATE_COL]).iloc[64:].reset_index(drop=True)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), dates.loc[valid].reset_index(drop=True)


def fit_elasticnet_base(train_df: pd.DataFrame, test_df: pd.DataFrame, target_mode: str = "return") -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series, pd.Series]:
    train_date_col = infer_date_col(train_df)
    test_date_col = infer_date_col(test_df)
    train_exog = train_df.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_df.copy()
    combined = pd.concat([train_exog, test_exog], ignore_index=True)
    combined = engineer_exog(combined, train_date_col)
    if target_mode == "return":
        params = dict(alpha=0.006579332246575682, l1_ratio=0.2)
    else:
        params = dict(alpha=0.01, l1_ratio=0.2)
    X_train, y_train, train_dates = build_supervised(train_df, combined.iloc[: len(train_df)].reset_index(drop=True), target_mode)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=50000, random_state=42, **params)),
    ])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    history = pd.to_numeric(train_df[TARGET_COL], errors="coerce").astype(float).tolist()
    test_preds = []
    for i in range(len(test_df)):
        idx = len(train_df) + i
        row = {c: combined.iloc[idx][c] for c in combined.columns if c != DATE_COL}
        row.update(target_context(history))
        X_row = pd.DataFrame([row]).reindex(columns=X_train.columns, fill_value=np.nan)
        pred = float(model.predict(X_row)[0])
        next_level = history[-1] * math.exp(pred) if target_mode == "return" else history[-1] + pred
        test_preds.append(next_level)
        history.append(next_level)
    return np.asarray(train_pred, dtype=float), np.asarray(test_preds, dtype=float), X_train, y_train, train_dates


def date_features(dates: Sequence[pd.Timestamp]) -> np.ndarray:
    dt = pd.to_datetime(pd.Series(dates))
    dow = dt.dt.dayofweek.to_numpy(dtype=float)
    month = dt.dt.month.to_numpy(dtype=float)
    doy = dt.dt.dayofyear.to_numpy(dtype=float)
    return np.column_stack([
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        doy / 365.25,
    ]).astype(np.float32)


def scale_1d(values: np.ndarray) -> tuple[np.ndarray, RobustScaler]:
    scaler = RobustScaler()
    scaled = scaler.fit_transform(np.asarray(values, dtype=float).reshape(-1, 1)).reshape(-1)
    return scaled.astype(np.float32), scaler


class LagTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.shape[1]
        h = self.in_proj(x) + self.pos[:, :l, :]
        mask = torch.triu(torch.ones(l, l, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=mask)
        h = self.norm(h[:, -1, :])
        return self.head(h).squeeze(-1)


def build_sequence_table(
    residuals: np.ndarray,
    backbone: np.ndarray,
    dates: Sequence[pd.Timestamp],
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    res_s, res_scaler = scale_1d(residuals)
    back_s, back_scaler = scale_1d(backbone)
    back_diff_s, diff_scaler = scale_1d(np.diff(backbone, prepend=backbone[0]))
    feats = date_features(dates)
    xs = []
    ys = []
    for t in range(context_len, len(res_s)):
        seq = np.concatenate([
            np.column_stack([res_s[t-context_len:t], back_s[t-context_len:t], back_diff_s[t-context_len:t], feats[t-context_len:t]]),
            np.column_stack([np.zeros(1, dtype=np.float32), back_s[t:t+1], back_diff_s[t:t+1], feats[t:t+1]]),
        ], axis=0)
        xs.append(seq.astype(np.float32))
        ys.append(float(res_s[t]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), res_scaler, back_scaler, diff_scaler


def train_lag_transformer(
    residuals: np.ndarray,
    backbone: np.ndarray,
    dates: Sequence[pd.Timestamp],
    context_len: int,
    epochs: int = 20,
) -> tuple[LagTransformer, dict[str, RobustScaler]]:
    X, y, res_scaler, back_scaler, diff_scaler = build_sequence_table(residuals, backbone, dates, context_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = LagTransformer(input_dim=X.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    split = max(int(len(X_t) * 0.85), 1)
    X_tr, y_tr = X_t[:split], y_t[:split]
    X_va, y_va = X_t[split:], y_t[split:]
    best_state = None
    best_loss = float("inf")
    patience = 5
    patience_left = patience
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for s in range(0, len(perm), 256):
            idx = perm[s:s+256]
            xb = X_tr[idx].to(DEVICE)
            yb = y_tr[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.smooth_l1_loss(model(X_va.to(DEVICE)), y_va.to(DEVICE)).cpu().item()) if len(X_va) else float("inf")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"res": res_scaler, "back": back_scaler, "diff": diff_scaler}


def forecast_residuals(
    model: LagTransformer,
    scalers: dict[str, RobustScaler],
    residual_hist: List[float],
    backbone_hist: List[float],
    date_hist: List[pd.Timestamp],
    future_backbone: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    context_len: int,
) -> np.ndarray:
    res_scaler = scalers["res"]
    back_scaler = scalers["back"]
    diff_scaler = scalers["diff"]
    out = []
    h_res = list(residual_hist)
    h_back = list(backbone_hist)
    h_dates = list(date_hist)
    model.eval()
    for d, back in zip(future_dates, future_backbone):
        h_back_tail = np.asarray(h_back[-context_len:], dtype=np.float32)
        h_res_tail = np.asarray(h_res[-context_len:], dtype=np.float32)
        h_diff_tail = np.diff(h_back_tail, prepend=h_back_tail[0])
        seq = np.concatenate([
            np.column_stack([
                res_scaler.transform(h_res_tail.reshape(-1, 1)).reshape(-1),
                back_scaler.transform(h_back_tail.reshape(-1, 1)).reshape(-1),
                diff_scaler.transform(h_diff_tail.reshape(-1, 1)).reshape(-1),
                date_features(h_dates[-context_len:]),
            ]),
            np.column_stack([
                np.zeros(1, dtype=np.float32),
                back_scaler.transform(np.array([[back]], dtype=np.float32)).reshape(-1),
                diff_scaler.transform(np.array([[back - h_back[-1]]], dtype=np.float32)).reshape(-1),
                date_features([d]),
            ]),
        ], axis=0)
        xb = torch.tensor(seq[None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_s = float(model(xb).item())
        pred = float(res_scaler.inverse_transform(np.array([[pred_s]], dtype=np.float32))[0, 0])
        out.append(pred)
        h_res.append(pred)
        h_back.append(float(back))
        h_dates.append(d)
    return np.asarray(out, dtype=float)


def pure_lag_transformer(train_df: pd.DataFrame, test_df: pd.DataFrame, context_len: int, target_mode: str = "return") -> np.ndarray:
    levels = pd.to_numeric(train_df[TARGET_COL], errors="coerce").astype(float).tolist()
    if target_mode == "return":
        series = np.array([0.0] + [math.log(levels[i] / levels[i - 1]) for i in range(1, len(levels))], dtype=float)
    else:
        series = np.array([np.nan] + [levels[i] - levels[i - 1] for i in range(1, len(levels))], dtype=float)
    series = series[1:]
    dates = pd.to_datetime(train_df[DATE_COL]).iloc[1:].tolist()
    scaled, scaler = scale_1d(series)
    feats = date_features(dates)
    xs = []
    ys = []
    for t in range(context_len, len(scaled)):
        seq = np.concatenate([
            np.column_stack([scaled[t-context_len:t], feats[t-context_len:t]]),
            np.column_stack([np.zeros(1, dtype=np.float32), feats[t:t+1]]),
        ], axis=0)
        xs.append(seq.astype(np.float32))
        ys.append(float(scaled[t]))
    X = torch.tensor(np.asarray(xs, dtype=np.float32))
    y = torch.tensor(np.asarray(ys, dtype=np.float32), dtype=torch.float32)
    model = LagTransformer(input_dim=X.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    split = max(int(len(X) * 0.85), 1)
    best_state = None
    best_loss = float("inf")
    patience = 5
    patience_left = patience
    for _ in range(20):
        model.train()
        perm = torch.randperm(len(X[:split]))
        for s in range(0, len(perm), 256):
            idx = perm[s:s+256]
            xb = X[:split][idx].to(DEVICE)
            yb = y[:split][idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            if len(X[split:]):
                val_loss = float(nn.functional.smooth_l1_loss(model(X[split:].to(DEVICE)), y[split:].to(DEVICE)).cpu().item())
            else:
                val_loss = float("inf")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Forecast recursively
    hist = series.tolist()
    hist_dates = pd.to_datetime(train_df[DATE_COL]).iloc[1:].tolist()
    preds = []
    for d in pd.to_datetime(test_df[DATE_COL]).tolist():
        ctx = np.asarray(hist[-context_len:], dtype=np.float32)
        ctx_s = scaler.transform(ctx.reshape(-1, 1)).reshape(-1)
        seq = np.concatenate([
            np.column_stack([ctx_s, date_features(hist_dates[-context_len:])]),
            np.column_stack([np.zeros(1, dtype=np.float32), date_features([d])]),
        ], axis=0)
        xb = torch.tensor(seq[None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_s = float(model(xb).item())
        pred = float(scaler.inverse_transform(np.array([[pred_s]], dtype=np.float32))[0, 0])
        last_level = levels[-1] if not preds else preds[-1]
        next_level = last_level * math.exp(pred) if target_mode == "return" else last_level + pred
        preds.append(next_level)
        hist.append(pred)
        hist_dates.append(d)
    return np.asarray(preds, dtype=float)


def main() -> None:
    seed_everything(42)
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    y_true = pd.to_numeric(actual[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    # ElasticNet baseline
    base_train_pred, base_test_pred, X_train, y_train, train_dates = fit_elasticnet_base(train, test, target_mode="return")
    base_rmse = rmse(y_true, base_test_pred)

    residual = y_train.to_numpy(dtype=float) - base_train_pred
    train_levels = pd.to_numeric(train[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    train_logret = np.array([np.nan] + [math.log(train_levels[i] / train_levels[i - 1]) for i in range(1, len(train_levels))], dtype=float)[1:]
    train_dates = train_dates.tolist()

    sweep_rows = []

    # Pure lag transformer
    for ctx in [64, 128]:
        preds = pure_lag_transformer(train, test, context_len=ctx, target_mode="return")
        sweep_rows.append({"model": "pure_lag_transformer_return", "context_len": ctx, "rmse": rmse(y_true, preds), "mae": float(mean_absolute_error(y_true, preds)), "mape": mape(y_true, preds)})

    # ElasticNet residual lag transformer
    # Build backbone history from in-sample fitted values on return space and recursive test predictions.
    backbone_train = np.asarray(base_train_pred, dtype=float)
    for ctx in [32, 64, 128]:
        model, scalers = train_lag_transformer(residual, backbone_train, train_dates, context_len=ctx, epochs=20)
        future_backbone = []
        hist_levels = train_levels.tolist()
        # reconstruct backbone test prediction in return space using same recursive process as base model
        # via test predictions already computed
        for v in base_test_pred:
            future_backbone.append(v)
        resid_pred = forecast_residuals(model, scalers, residual_hist=residual.tolist(), backbone_hist=backbone_train.tolist(), date_hist=train_dates, future_backbone=future_backbone, future_dates=pd.to_datetime(test[DATE_COL]).tolist(), context_len=ctx)
        final_preds = base_test_pred + resid_pred
        sweep_rows.append({"model": "elasticnet_plus_lag_residual", "context_len": ctx, "rmse": rmse(y_true, final_preds), "mae": float(mean_absolute_error(y_true, final_preds)), "mape": mape(y_true, final_preds)})

    results = pd.DataFrame(sweep_rows).sort_values("rmse").reset_index(drop=True)
    results.to_csv("lag_transformer_sweep_results.csv", index=False)
    report = [
        "# Lag Transformer Sweep",
        "",
        f"ElasticNet baseline RMSE: `{base_rmse:.2f}`",
        "",
        "| model | context | RMSE | MAE | MAPE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in results.iterrows():
        report.append(f"| `{r['model']}` | `{int(r['context_len'])}` | `{r['rmse']:.2f}` | `{r['mae']:.2f}` | `{r['mape']:.2f}%` |")
    report.append("")
    report.append("Best result is the first row of the table.")
    Path("lag_transformer_sweep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
