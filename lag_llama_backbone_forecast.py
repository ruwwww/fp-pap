#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_CSV = ROOT / "data_test.csv"
ACTUAL_CSV = ROOT / "data_test_actual.csv"
DATE_COL = "Date"
TARGET_COL = "USDIDR"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def date_features(dates: Sequence[pd.Timestamp]) -> np.ndarray:
    dt = pd.to_datetime(pd.Series(dates))
    dow = dt.dt.dayofweek.to_numpy(dtype=float)
    month = dt.dt.month.to_numpy(dtype=float)
    year_progress = (dt.dt.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    out = np.column_stack([
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * month / 12.0),
        np.cos(2 * np.pi * month / 12.0),
        year_progress,
    ])
    return out.astype(np.float32)


def gaussian_nll(mu: torch.Tensor, scale_raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = torch.nn.functional.softplus(scale_raw) + 1e-4
    z = (target - mu) / scale
    return (0.5 * z.pow(2) + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)).mean()


def scale_1d(values: np.ndarray) -> tuple[np.ndarray, RobustScaler]:
    scaler = RobustScaler()
    scaled = scaler.fit_transform(np.asarray(values, dtype=float).reshape(-1, 1)).reshape(-1)
    return scaled.astype(np.float32), scaler


class LagTokenTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        h = self.in_proj(x) + self.pos_emb[:, :seq_len, :]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(h, mask=mask)
        h = self.norm(h[:, -1, :])
        out = self.head(h)
        mu, scale_raw = out[:, 0], out[:, 1]
        return mu, scale_raw


@dataclass
class FitResult:
    context_len: int
    val_rmse: float
    val_mae: float
    val_mape: float
    model: LagTokenTransformer
    scalers: dict[str, RobustScaler]
    base_model: object


def build_samples(
    residuals: np.ndarray,
    base_levels: np.ndarray,
    base_deltas: np.ndarray,
    dates: Sequence[pd.Timestamp],
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    feats = date_features(dates)
    xs = []
    ys = []
    for t in range(context_len, len(residuals)):
        hist_res = residuals[t - context_len : t]
        hist_base = base_levels[t - context_len : t]
        hist_delta = base_deltas[t - context_len : t]
        hist_feat = feats[t - context_len : t]
        target_feat = feats[t : t + 1]
        seq = np.concatenate([
            np.column_stack([hist_res, hist_base, hist_delta, hist_feat]),
            np.column_stack([np.zeros(1, dtype=np.float32), base_levels[t : t + 1], base_deltas[t : t + 1], target_feat]),
        ], axis=0)
        xs.append(seq.astype(np.float32))
        ys.append(float(residuals[t]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_transformer(
    residuals: np.ndarray,
    base_levels: np.ndarray,
    dates: Sequence[pd.Timestamp],
    context_len: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
) -> tuple[LagTokenTransformer, dict[str, RobustScaler]]:
    residuals_scaled, residual_scaler = scale_1d(residuals)
    base_scaled, base_scaler = scale_1d(base_levels)
    base_delta_scaled, delta_scaler = scale_1d(np.diff(base_levels, prepend=base_levels[0]))
    X, y = build_samples(residuals_scaled, base_scaled, base_delta_scaled, dates, context_len)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    model = LagTokenTransformer(input_dim=X.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_t)
    split = max(int(n * 0.85), 1)
    X_tr, y_tr = X_t[:split], y_t[:split]
    X_va, y_va = X_t[split:], y_t[split:]

    best_state = None
    best_loss = float("inf")
    patience = 6
    patience_left = patience

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr[idx].to(DEVICE)
            yb = y_tr[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            mu, scale_raw = model(xb)
            loss = gaussian_nll(mu, scale_raw, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            if len(X_va) > 0:
                xb = X_va.to(DEVICE)
                yb = y_va.to(DEVICE)
                mu, scale_raw = model(xb)
                val_loss = float(gaussian_nll(mu, scale_raw, yb).cpu().item())
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
    return model, {"residual": residual_scaler, "base": base_scaler, "delta": delta_scaler}


def recursive_residual_forecast(
    model: LagTokenTransformer,
    scalers: dict[str, RobustScaler],
    residual_history: List[float],
    base_history: List[float],
    date_history: List[pd.Timestamp],
    future_base: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    context_len: int,
) -> np.ndarray:
    model.eval()
    preds = []
    hist_res = list(map(float, residual_history))
    hist_base = list(map(float, base_history))
    hist_dates = list(date_history)
    residual_scaler = scalers["residual"]
    base_scaler = scalers["base"]
    delta_scaler = scalers["delta"]
    for dt, base_val in zip(future_dates, future_base):
        base_delta = float(base_val - hist_base[-1]) if hist_base else 0.0
        res_ctx = np.asarray(hist_res[-context_len:], dtype=np.float32)
        base_ctx = np.asarray(hist_base[-context_len:], dtype=np.float32)
        delta_ctx = np.diff(base_ctx, prepend=base_ctx[0])
        date_ctx = date_features(hist_dates[-context_len:])
        target_feat = date_features([dt])
        seq = np.concatenate([
            np.column_stack([
                residual_scaler.transform(res_ctx.reshape(-1, 1)).reshape(-1),
                base_scaler.transform(base_ctx.reshape(-1, 1)).reshape(-1),
                delta_scaler.transform(delta_ctx.reshape(-1, 1)).reshape(-1),
                date_ctx,
            ]),
            np.column_stack([
                np.zeros(1, dtype=np.float32),
                base_scaler.transform(np.array([[base_val]], dtype=np.float32)).reshape(-1),
                delta_scaler.transform(np.array([[base_delta]], dtype=np.float32)).reshape(-1),
                target_feat,
            ]),
        ], axis=0)
        xb = torch.tensor(seq[None, :, :], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mu, _ = model(xb)
        pred_scaled = float(mu.item())
        pred = float(residual_scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32))[0, 0])
        preds.append(pred)
        hist_res.append(pred)
        hist_base.append(float(base_val))
        hist_dates.append(dt)
    return np.asarray(preds, dtype=float)


def fit_base_model(levels: pd.Series) -> object:
    mod = UnobservedComponents(levels.astype(float), level="local linear trend")
    res = mod.fit(disp=False)
    return res


def base_forecast(base_model, steps: int) -> np.ndarray:
    return np.asarray(base_model.get_forecast(steps=steps).predicted_mean, dtype=float)


def evaluate_hybrid(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    context_len: int,
    epochs: int,
) -> FitResult:
    train_levels = pd.to_numeric(train_df[TARGET_COL], errors="coerce").reset_index(drop=True)
    val_levels = pd.to_numeric(val_df[TARGET_COL], errors="coerce").reset_index(drop=True)
    base = fit_base_model(train_levels)
    train_base_fit = pd.Series(base.fittedvalues).reset_index(drop=True)
    train_resid_all = (train_levels - train_base_fit).to_numpy(dtype=float)
    mask = np.isfinite(train_resid_all)
    train_resid = train_resid_all[mask]
    train_base = train_base_fit.to_numpy(dtype=float)[mask]
    train_dates = pd.to_datetime(train_df[DATE_COL]).reset_index(drop=True).loc[mask].tolist()

    model, scalers = fit_transformer(train_resid, train_base, train_dates, context_len=context_len, epochs=epochs)
    val_base = base_forecast(base, len(val_df))
    val_resid = recursive_residual_forecast(
        model,
        scalers,
        residual_history=train_resid.tolist(),
        base_history=train_base.tolist(),
        date_history=train_dates,
        future_base=val_base,
        future_dates=pd.to_datetime(val_df[DATE_COL]).tolist(),
        context_len=context_len,
    )
    val_pred = val_base + val_resid
    return FitResult(
        context_len=context_len,
        val_rmse=rmse(val_levels, val_pred),
        val_mae=float(mean_absolute_error(val_levels, val_pred)),
        val_mape=mape(val_levels, val_pred),
        model=model,
        scalers=scalers,
        base_model=base,
    )


def forecast_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_len: int,
    epochs: int,
) -> np.ndarray:
    train_levels = pd.to_numeric(train_df[TARGET_COL], errors="coerce").reset_index(drop=True)
    train_dates_all = pd.to_datetime(train_df[DATE_COL]).reset_index(drop=True)
    base = fit_base_model(train_levels)
    train_base_fit = pd.Series(base.fittedvalues).reset_index(drop=True)
    train_resid_all = (train_levels - train_base_fit).to_numpy(dtype=float)
    mask = np.isfinite(train_resid_all)
    train_resid = train_resid_all[mask]
    train_base = train_base_fit.to_numpy(dtype=float)[mask]
    train_dates = train_dates_all.loc[mask].tolist()

    model, scalers = fit_transformer(train_resid, train_base, train_dates, context_len=context_len, epochs=epochs)
    base_pred = base_forecast(base, len(test_df))
    resid_pred = recursive_residual_forecast(
        model,
        scalers,
        residual_history=train_resid.tolist(),
        base_history=train_base.tolist(),
        date_history=train_dates,
        future_base=base_pred,
        future_dates=pd.to_datetime(test_df[DATE_COL]).tolist(),
        context_len=context_len,
    )
    return base_pred + resid_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Lag-Llama-style AR backbone benchmark")
    parser.add_argument("--train_csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test_csv", type=Path, default=TEST_CSV)
    parser.add_argument("--actual_csv", type=Path, default=ACTUAL_CSV)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_days", type=int, default=252)
    parser.add_argument("--output_csv", type=Path, default=Path("lag_llama_oos_predictions.csv"))
    parser.add_argument("--report_md", type=Path, default=Path("lag_llama_oos_report.md"))
    args = parser.parse_args()

    seed_everything(42)
    train = pd.read_csv(args.train_csv, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(args.test_csv, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(args.actual_csv, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    val_days = min(args.val_days, max(120, len(train) // 5))
    fit_df = train.iloc[:-val_days].reset_index(drop=True)
    val_df = train.iloc[-val_days:].reset_index(drop=True)

    candidates = [32, 64, 128]
    fits: list[FitResult] = []
    for c in candidates:
        fits.append(evaluate_hybrid(fit_df, val_df, context_len=c, epochs=args.epochs))

    best = sorted(fits, key=lambda x: x.val_rmse)[0]
    final_preds = forecast_test(train, test, context_len=best.context_len, epochs=args.epochs)

    actual_y = pd.to_numeric(actual[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    metrics = {
        "model": "LagLlama_AR_backbone_hybrid",
        "context_len": best.context_len,
        "val_rmse": best.val_rmse,
        "test_rmse": rmse(actual_y, final_preds),
        "test_mae": float(mean_absolute_error(actual_y, final_preds)),
        "test_mape": mape(actual_y, final_preds),
        "best_baseline_rmse": 588.9682104499586,
        "best_ssm_rmse": 588.9682104499586,
    }

    pred_df = pd.DataFrame({DATE_COL: test[DATE_COL], "USDIDR_pred": final_preds, "USDIDR_actual": actual_y})
    pred_df.to_csv(args.output_csv, index=False)

    report = [
        "# Lag-Llama Backbone Report",
        "",
        f"Best context length: `{best.context_len}`",
        f"Validation RMSE: `{best.val_rmse:.2f}`",
        "",
        "## OOS Test",
        f"RMSE: `{metrics['test_rmse']:.2f}`",
        f"MAE: `{metrics['test_mae']:.2f}`",
        f"MAPE: `{metrics['test_mape']:.2f}%`",
        "",
        "## Validation Sweep",
        "| context | val_rmse | val_mae | val_mape |",
        "| --- | ---: | ---: | ---: |",
    ]
    for f in fits:
        report.append(f"| `{f.context_len}` | `{f.val_rmse:.2f}` | `{f.val_mae:.2f}` | `{f.val_mape:.2f}%` |")
    report += [
        "",
        "## Benchmark",
        "- Best prior SSM: `RMSE 588.97` (`MarkovAR_4reg_p1`)",
        "- This run uses a local linear trend backbone plus a lag-token transformer residual corrector.",
        "- No direct exogenous predictors were used.",
    ]
    args.report_md.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Saved predictions to {args.output_csv}")
    print(f"Saved report to {args.report_md}")


if __name__ == "__main__":
    main()
