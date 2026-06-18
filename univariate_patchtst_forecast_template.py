#!/usr/bin/env python3
"""PatchTST Multivariate Forecasting Pipeline with Future Covariate Overwrite.

Assumptions:
- Train has the full feature set including the target column USDIDR.
- Test has all non-target features available for the entire forecast horizon.
- No engineered features are used in this version.

Strategy:
1) Train a multivariate PatchTST on raw features.
2) At inference, predict all channels for each rolling horizon.
3) Overwrite non-target channels with the known future covariates from test.
4) Keep only USDIDR predictions from the model output.

Notes:
- This uses Hugging Face transformers' PatchTSTForPrediction API.
- Training uses outputs.loss from the model, not manual MSE.
- Scaling is handled manually with RobustScaler statistics so test covariates
  can be transformed without requiring the missing target column.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTForPrediction

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TARGET_COL = "USDIDR"

# Raw multivariate features only (no engineered features in this version)
FEATURE_COLS: List[str] = [
    "USDIDR",
    "OIL",
    "GOLD",
    "SP500",
    "IHSG",
    "VIX",
    "CPI",
    "BI_rate",
    "US_rate",
]

RANDOM_STATE = 42
LOOKBACK_WINDOW = 784   # ~2 years of trading days
ROLLING_HORIZON = 21    # ~1 month of trading days per roll step
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 9.561684653393984e-04

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    return candidates[0] if candidates else None


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_and_align_dataframe(
    df: pd.DataFrame,
    *,
    required_cols: Sequence[str],
    date_col: str,
    is_train: bool,
) -> pd.DataFrame:
    """
    Keep only required columns plus date, coerce to numeric, and repair missing values.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[[date_col] + list(required_cols)].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    for c in required_cols:
        out[c] = _safe_numeric_series(out[c])

    out = out.sort_values(date_col).reset_index(drop=True)

    # Simple repair for gaps
    out[required_cols] = out[required_cols].interpolate(limit_direction="both")
    out[required_cols] = out[required_cols].ffill().bfill()

    # Light anomaly handling, retained from your template
    if TARGET_COL in out.columns:
        out.loc[out[TARGET_COL] < 1000, TARGET_COL] *= 10

    if "OIL" in out.columns:
        out["OIL"] = out["OIL"].clip(lower=0)

    if "VIX" in out.columns:
        # Cap extreme spikes while preserving tail behavior
        if out["VIX"].notna().any():
            out["VIX"] = out["VIX"].clip(upper=out["VIX"].quantile(0.99))

    if is_train:
        # No special train-only transformation here; placeholder for symmetry/readability.
        pass

    return out


def robust_transform_with_stats(values: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Apply RobustScaler transform using precomputed center_ and scale_.
    Works even if the target column is unavailable in test.
    """
    scale_safe = np.where(scale == 0, 1.0, scale)
    return (values - center) / scale_safe


def robust_inverse_transform_target(target_scaled: np.ndarray, center: float, scale: float) -> np.ndarray:
    scale_safe = 1.0 if scale == 0 else float(scale)
    return target_scaled * scale_safe + float(center)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class TimeSeriesDataset(Dataset):
    """Windowed multivariate dataset for PatchTST."""

    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        n = len(self.data) - self.lookback - self.horizon + 1
        if n <= 0:
            raise ValueError(
                f"Dataset kosong! Data ({len(self.data)} baris) tidak cukup untuk "
                f"lookback={lookback} + horizon={horizon}. "
                f"Butuh minimal {lookback + horizon} baris."
            )

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.lookback]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon]
        return x, y


# -----------------------------------------------------------------------------
# Rolling inference
# -----------------------------------------------------------------------------
def rolling_forecast_with_future_covariates(
    model: PatchTSTForPrediction,
    seed_context: np.ndarray,                 # (LOOKBACK_WINDOW, C)
    future_covariates_scaled: np.ndarray,     # (total_steps, C-1)
    total_steps: int,
    step_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Rolling forecast where the target is autoregressive, but future covariates
    are overwritten with the known test values at each roll.
    """
    model.eval()

    if future_covariates_scaled.shape[0] < total_steps:
        raise ValueError(
            f"future_covariates_scaled has {future_covariates_scaled.shape[0]} rows, "
            f"but total_steps={total_steps}."
        )

    buffer = seed_context.copy().tolist()  # list of [C] rows
    all_preds: list[np.ndarray] = []

    n_rolls = math.ceil(total_steps / step_size)
    print(f"Rolling Forecast: {total_steps} hari → {n_rolls} roll-steps × {step_size} hari/step")

    with torch.no_grad():
        for roll in range(n_rolls):
            start = roll * step_size
            end = min(start + step_size, total_steps)
            this_step = end - start

            context = np.array(buffer[-LOOKBACK_WINDOW:], dtype=np.float32)  # (L, C)
            ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, C)

            out = model(past_values=ctx_tensor)
            preds = out.prediction_outputs.squeeze(0).detach().cpu().numpy()  # (prediction_length, C)

            # Keep only the portion we need for this roll
            preds = preds[:this_step].copy()

            # Overwrite non-target channels with known future covariates
            preds[:, 1:] = future_covariates_scaled[start:end]

            all_preds.append(preds)
            buffer.extend(preds.tolist())

            steps_done = end
            print(f"  Roll [{roll + 1:03d}/{n_rolls}] — {steps_done}/{total_steps} hari selesai")

    full_preds = np.concatenate(all_preds, axis=0)[:total_steps]  # (total_steps, C)
    return full_preds


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    submission_csv: Path,
    actual_test_csv: Optional[Path] = None,
    lookback_window: int = LOOKBACK_WINDOW,
    rolling_horizon: int = ROLLING_HORIZON,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> None:
    print(f"Device          : {DEVICE}")
    print(f"Lookback Window  : {lookback_window} hari")
    print(f"Rolling Horizon  : {rolling_horizon} hari/step")
    print(f"Feature Columns  : {FEATURE_COLS}")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 1) Load
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_raw = pd.read_csv(actual_test_csv) if actual_test_csv else None

    date_col = infer_date_col(train_raw)
    if not date_col:
        raise ValueError("Kolom tanggal tidak ditemukan di train_csv.")

    # Date cols in test/actual should match
    test_date_col = infer_date_col(test_raw)
    if not test_date_col:
        raise ValueError("Kolom tanggal tidak ditemukan di test_csv.")
    if test_date_col != date_col:
        # This is fine if the name differs but semantics are same; we normalize below.
        pass

    train_raw = clean_and_align_dataframe(train_raw, required_cols=FEATURE_COLS, date_col=date_col, is_train=True)
    test_raw = clean_and_align_dataframe(test_raw, required_cols=[c for c in FEATURE_COLS if c != TARGET_COL],
                                          date_col=test_date_col, is_train=False)

    if actual_raw is not None:
        actual_raw = pd.read_csv(actual_test_csv)

    # Validate train/test coverage
    missing_test_covs = [c for c in FEATURE_COLS[1:] if c not in test_raw.columns]
    if missing_test_covs:
        raise ValueError(f"Test harus memiliki covariates berikut: {missing_test_covs}")

    total_test_steps = len(test_raw)
    n_rolls = math.ceil(total_test_steps / rolling_horizon)
    print(f"\nTrain rows      : {len(train_raw)}")
    print(f"Test rows       : {total_test_steps} hari → {n_rolls} roll-steps")

    # 2) Scaling on train only
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_raw[FEATURE_COLS].values.astype(np.float32))
    center = scaler.center_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)

    # 3) Transform test covariates using train scaler stats
    test_cov_values = test_raw[FEATURE_COLS[1:]].values.astype(np.float32)
    test_cov_scaled = robust_transform_with_stats(test_cov_values, center[1:], scale[1:]).astype(np.float32)

    # 4) Build training dataset / loader
    train_dataset = TimeSeriesDataset(train_scaled, lookback_window, rolling_horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Training windows: {len(train_dataset)}")

    # 5) Model config
    config = PatchTSTConfig(
        num_input_channels=len(FEATURE_COLS),
        context_length=lookback_window,
        prediction_length=rolling_horizon,
        patch_length=16,
        patch_stride=8,
        num_hidden_layers=3,
        d_model=256,
        num_attention_heads=8,
        ffn_dim=512,
        share_embedding=True,
        channel_attention=False,
        pooling_type="mean",
        scaling=None,   # scaling handled externally with RobustScaler
    )

    model = PatchTSTForPrediction(config).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 60)
    print(" Training PatchTST — Multivariate + Future Covariate Overwrite")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for past_v, future_v in train_loader:
            past_v = past_v.to(DEVICE)      # (B, L, C)
            future_v = future_v.to(DEVICE)  # (B, H, C)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(past_values=past_v, future_values=future_v)
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Model did not return loss during training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * past_v.size(0)

        scheduler.step()
        epoch_loss = total_loss / len(train_dataset)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {epoch_loss:.6f} | LR: {current_lr:.2e}")

    # 6) Rolling inference
    print("\n" + "=" * 60)
    print(" Mulai Rolling Forecast")
    print("=" * 60)

    seed_context = train_scaled[-lookback_window:]  # (L, C)
    pred_scaled_full = rolling_forecast_with_future_covariates(
        model=model,
        seed_context=seed_context,
        future_covariates_scaled=test_cov_scaled,
        total_steps=total_test_steps,
        step_size=rolling_horizon,
        device=DEVICE,
    )  # (T, C)

    # 7) Inverse transform only target channel
    target_scaled = pred_scaled_full[:, 0]
    final_preds = robust_inverse_transform_target(target_scaled, center[0], scale[0])

    # 8) Save submission
    submission = pd.DataFrame(
        {
            "Date": pd.to_datetime(test_raw[test_date_col]).dt.strftime("%Y-%m-%d"),
            TARGET_COL: final_preds,
        }
    )
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"\nSubmission disimpan ke: {submission_csv}")

    # 9) Optional evaluation
    if actual_raw is not None:
        if TARGET_COL not in actual_raw.columns:
            raise ValueError(f"actual_test_csv harus memiliki kolom {TARGET_COL}.")
        actual_values = pd.to_numeric(actual_raw[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)
        if len(actual_values) != len(final_preds):
            raise ValueError(
                f"Panjang actual_test_csv ({len(actual_values)}) tidak sama dengan prediksi ({len(final_preds)})."
            )

        rmse_model = float(np.sqrt(mean_squared_error(actual_values, final_preds)))
        rmse_naive = float(np.sqrt(mean_squared_error(actual_values[1:], actual_values[:-1])))
        skill = (rmse_naive - rmse_model) / rmse_naive * 100

        print(f"\n{'=' * 60}")
        print(f" EVALUASI BENCHMARK")
        print(f"{'=' * 60}")
        print(f" Naive Baseline RMSE (Yesterday=Today) : {rmse_naive:.4f}")
        print(f" PatchTST Rolling RMSE                 : {rmse_model:.4f}")
        print(f" Skill Score vs Naive                  : {skill:+.2f}%")
        print(f"{'=' * 60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PatchTST Multivariate Rolling Forecast Pipeline")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--submission_csv", type=Path, required=True)
    p.add_argument("--actual_test_csv", type=Path, default=None)
    p.add_argument("--lookback_window", type=int, default=LOOKBACK_WINDOW)
    p.add_argument("--rolling_horizon", type=int, default=ROLLING_HORIZON)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        submission_csv=args.submission_csv,
        actual_test_csv=args.actual_test_csv,
        lookback_window=args.lookback_window,
        rolling_horizon=args.rolling_horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
