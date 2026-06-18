#!/usr/bin/env python3
"""PatchTST Hyperparameter Tuning Pipeline using Optuna & Hugging Face.

Requirements:
    pip install optuna
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from transformers import PatchTSTConfig, PatchTSTForPrediction

# Optional but recommended for clean tuning output
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

TARGET_COL = "USDIDR"
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Dataset & Preprocessing Utilities
# ──────────────────────────────────────────────
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates: return candidates[0]
    return None

def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TARGET_COL in out.columns:
        out.loc[out[TARGET_COL] < 1000, TARGET_COL] *= 10
    if "OIL" in out.columns:
        out["OIL"] = out["OIL"].clip(lower=0)
    if "VIX" in out.columns:
        cap = out["VIX"].quantile(0.99)
        out["VIX"] = out["VIX"].clip(upper=cap)
    return out

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past_values = self.data[idx : idx + self.lookback]
        future_values = self.data[idx + self.lookback : idx + self.lookback + self.horizon]
        return past_values, future_values

# ──────────────────────────────────────────────
# Hyperparameter Search & Training
# ──────────────────────────────────────────────
def run_tuning(train_csv: Path, test_csv: Path, n_trials: int = 10) -> None:
    print(f"Menggunakan Device: {DEVICE} untuk Hyperparameter Search")
    torch.manual_seed(RANDOM_STATE)

    # Load & Clean
    train_raw = fix_anomalies(pd.read_csv(train_csv))
    test_raw = pd.read_csv(test_csv)
    date_col = infer_date_col(train_raw)
    
    train_raw = train_raw.sort_values(date_col).reset_index(drop=True)
    test_raw = test_raw.sort_values(date_col).reset_index(drop=True)
    horizon = len(test_raw)

    feature_order = [TARGET_COL]
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_raw[feature_order].values)

    # Objective function untuk Optuna
    def objective(trial: optuna.Trial) -> float:
        # 1. Tentukan ruang pencarian (Search Space)
        lookback_window = trial.suggest_categorical("lookback_window", [252, 378, 504])
        patch_length = trial.suggest_categorical("patch_length", [8, 16, 20, 24])
        stride = trial.suggest_categorical("stride", [4, 8, 10, 12])
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 10, 20)
        
        # Validasi constraint: stride tidak boleh lebih besar dari patch_length
        if stride > patch_length:
            raise optuna.TrialPruned()

        # 2. Re-create Dataloader berdasarkan lookback_window yang dipilih trial ini
        train_dataset = TimeSeriesDataset(train_scaled, lookback_window, horizon)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 3. Init Model
        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=lookback_window,
            prediction_length=horizon,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            num_attention_heads=num_heads,
            encoder_layers=3,
            scaling="std",
        )
        model = PatchTSTForPrediction(config).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        # 4. Pelatihan Singkat per Trial
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for past_v, future_v in train_loader:
                past_v, future_v = past_v.to(DEVICE), future_v.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(past_values=past_v, future_values=future_v)
                loss = criterion(outputs.prediction_outputs, future_v)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * past_v.size(0)
            
            current_epoch_loss = total_loss / len(train_dataset)
            
            # Mengizinkan Optuna memotong trial jika di tengah jalan loss-nya ampas
            trial.report(current_epoch_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return current_epoch_loss

    print("\n" + "="*50)
    print(f" Memulai Pencarian Berbasis Bayesian (Total {n_trials} Trials)...")
    print("="*50)

    # Gunakan TPE Sampler (Bayesian Optimization)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "="*50)
    print(" TUNING SELESAI! HYPERPARAMETER TERBAIK:")
    print("="*50)
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")
    print(f" - Best In-sample Train Loss: {study.best_value:.6f}")
    print("="*50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna Hyperparam Search for PatchTST")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--n_trials", type=int, default=15, help="Jumlah percobaan tuning")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_tuning(train_csv=args.train_csv, test_csv=args.test_csv, n_trials=args.n_trials)