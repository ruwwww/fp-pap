#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

import lag_transformer_sweep as lts

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"
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
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def date_features(dates) -> np.ndarray:
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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return train, test, actual


def build_train_table(levels: np.ndarray, dates, context_len: int) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    logret = np.array([math.log(levels[i] / levels[i - 1]) for i in range(1, len(levels))], dtype=float)
    scaled, scaler = scale_1d(logret)
    feats = date_features(dates[1:])
    xs = []
    ys = []
    for t in range(context_len, len(scaled)):
        seq = np.concatenate([
            np.column_stack([scaled[t - context_len:t], feats[t - context_len:t]]),
            np.column_stack([np.zeros(1, dtype=np.float32), feats[t:t + 1]]),
        ], axis=0)
        xs.append(seq.astype(np.float32))
        ys.append(float(scaled[t]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), scaler


def train_univariate_model(train: pd.DataFrame, context_len: int, seed: int = 5, epochs: int = 20) -> tuple[LagTransformer, RobustScaler]:
    seed_everything(seed)
    levels = pd.to_numeric(train[TARGET_COL], errors="coerce").astype(float).to_numpy(dtype=float)
    dates = pd.to_datetime(train[DATE_COL]).tolist()
    X, y, scaler = build_train_table(levels, dates, context_len)
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
            idx = perm[s:s + 256]
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
            if len(X_va):
                val_loss = float(nn.functional.smooth_l1_loss(model(X_va.to(DEVICE)), y_va.to(DEVICE)).cpu().item())
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
    return model, scaler


def forecast_univariate(model: LagTransformer, scaler: RobustScaler, train: pd.DataFrame, test: pd.DataFrame, context_len: int) -> np.ndarray:
    levels = pd.to_numeric(train[TARGET_COL], errors="coerce").astype(float).to_list()
    logret_hist = [math.log(levels[i] / levels[i - 1]) for i in range(1, len(levels))]
    hist_dates = pd.to_datetime(train[DATE_COL]).iloc[1:].tolist()
    preds = []
    model.eval()

    for d in pd.to_datetime(test[DATE_COL]).tolist():
        ctx = np.asarray(logret_hist[-context_len:], dtype=np.float32)
        ctx_s = scaler.transform(ctx.reshape(-1, 1)).reshape(-1)
        seq = np.concatenate([
            np.column_stack([ctx_s, date_features(hist_dates[-context_len:])]),
            np.column_stack([np.zeros(1, dtype=np.float32), date_features([d])]),
        ], axis=0)
        xb = torch.tensor(seq[None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred_s = float(model(xb).item())
        pred_ret = float(scaler.inverse_transform(np.array([[pred_s]], dtype=np.float32))[0, 0])
        last_level = levels[-1] if not preds else preds[-1]
        next_level = last_level * math.exp(pred_ret)
        preds.append(next_level)
        logret_hist.append(pred_ret)
        hist_dates.append(d)
    return np.asarray(preds, dtype=float)


def risk_gate(base_preds: np.ndarray, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    levels = pd.to_numeric(train[TARGET_COL], errors="coerce").astype(float).to_list()
    gated = []
    vix_test = pd.to_numeric(test["VIX"], errors="coerce").fillna(18.0).to_numpy(dtype=float)
    us_rate_test = pd.to_numeric(test["US_rate"], errors="coerce").fillna(4.5).to_numpy(dtype=float)
    bi_rate_test = pd.to_numeric(test["BI_rate"], errors="coerce").fillna(5.5).to_numpy(dtype=float)

    for t, pred_level in enumerate(base_preds):
        prev_level = levels[-1] if not gated else gated[-1]
        predicted_logret = math.log(pred_level / prev_level)
        current_vix = float(vix_test[t])
        current_spread = float(us_rate_test[t] - bi_rate_test[t])

        if current_vix < 15.0:
            predicted_logret *= 0.2
        if current_spread > -1.0 and predicted_logret < -0.002:
            predicted_logret = -0.0005
        predicted_logret = float(np.clip(predicted_logret, -0.015, 0.015))

        next_level = prev_level * math.exp(predicted_logret)
        gated.append(next_level)
    return np.asarray(gated, dtype=float)


def main() -> None:
    train, test, actual = load_data()
    y_true = pd.to_numeric(actual[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    context_len = 72
    seed = 5

    base_artifact = ROOT / "lag_transformer_best_predictions.csv"
    if base_artifact.exists():
        base_df = pd.read_csv(base_artifact, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
        base_preds = pd.to_numeric(base_df["USDIDR_pred"], errors="coerce").to_numpy(dtype=float)
    else:
        lts.seed_everything(seed)
        base_preds = lts.pure_lag_transformer(train, test, context_len=context_len, target_mode="return")
    gated_preds = risk_gate(base_preds, train, test)

    results = pd.DataFrame([
        {"model": "univariate_lag_transformer", "rmse": rmse(y_true, base_preds), "mae": float(mean_absolute_error(y_true, base_preds)), "mape": mape(y_true, base_preds)},
        {"model": "univariate_lag_transformer_gated", "rmse": rmse(y_true, gated_preds), "mae": float(mean_absolute_error(y_true, gated_preds)), "mape": mape(y_true, gated_preds)},
    ])
    results.to_csv("gated_univariate_verification_results.csv", index=False)

    pred_df = pd.DataFrame({
        DATE_COL: test[DATE_COL],
        "base_pred": base_preds,
        "gated_pred": gated_preds,
        "actual": y_true,
    })
    pred_df.to_csv("gated_univariate_predictions.csv", index=False)

    best = results.sort_values("rmse").iloc[0]
    report = [
        "# Gated Univariate Verification",
        "",
        f"Context length: `{context_len}`",
        f"Seed: `{seed}`",
        "",
        results.to_markdown(index=False),
        "",
        f"Best config: `{best['model']}`",
        f"Best RMSE: `{best['rmse']:.2f}`",
        "",
        "## Gate Rules",
        "- VIX < 15: damp predicted log-return by 80%.",
        "- If US_rate - BI_rate is tight and prediction implies strong Rupiah strengthening, clamp it.",
        "- Clip daily log-return to [-0.015, 0.015].",
        "",
        "## Interpretation",
        "- Core model remains univariate log-return only.",
        "- Macro is used only as an external risk governor.",
        "- This verifies whether gating can protect AR inertia without direct exogenous injection.",
    ]
    (ROOT / "gated_univariate_verification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
