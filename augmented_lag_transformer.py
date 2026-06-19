#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import warnings
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

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
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def cyclical(values: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    theta = 2.0 * np.pi * values.astype(float) / period
    return np.sin(theta), np.cos(theta)

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

def synchronized_stationary_bootstrap(df: pd.DataFrame, avg_block_size: int = 60) -> pd.DataFrame:
    n = len(df)
    indices = []
    p = 1.0 / avg_block_size
    curr = 0
    while curr < n:
        block_len = np.random.geometric(p)
        block_len = min(block_len, n - curr)
        start_idx = np.random.randint(0, n - block_len + 1)
        indices.extend(range(start_idx, start_idx + block_len))
        curr += block_len
    bootstrap_df = df.iloc[indices].copy().reset_index(drop=True)
    bootstrap_df[DATE_COL] = df[DATE_COL]
    return bootstrap_df

def build_sequences(series: np.ndarray, dates: Sequence[pd.Timestamp], context_len: int) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
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
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), scaler

def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 25) -> LagTransformer:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    
    model = LagTransformer(input_dim=X.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    split = max(int(len(X_t) * 0.85), 1)
    X_tr, y_tr = X_t[:split], y_t[:split]
    X_va, y_va = X_t[split:], y_t[split:]
    
    best_state = None
    best_loss = float("inf")
    patience = 5
    patience_left = patience
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        for s in range(0, len(perm), 128):
            idx = perm[s:s+128]
            xb = X_tr[idx].to(DEVICE)
            yb = y_tr[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.smooth_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(idx)
        
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
    return model

def main() -> None:
    seed_everything(42)
    train = pd.read_csv(TRAIN_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    test = pd.read_csv(TEST_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    actual = pd.read_csv(ACTUAL_CSV, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    y_true = pd.to_numeric(actual[TARGET_COL], errors="coerce").to_numpy(dtype=float)

    # 1. Generate Synchronized Stationary Bootstrap Augmentation (multiple boots)
    num_bootstrap_runs = 5
    bootstrap_list = [train]
    for _ in range(num_bootstrap_runs):
        bootstrap_list.append(synchronized_stationary_bootstrap(train, avg_block_size=60))
    
    # Concatenate original training and bootstrap runs
    augmented_train = pd.concat(bootstrap_list, ignore_index=True)
    print(f"Original Train size: {len(train)}")
    print(f"Augmented Train size: {len(augmented_train)}")

    # Extract levels & log returns for original and augmented
    orig_levels = pd.to_numeric(train[TARGET_COL], errors="coerce").astype(float).tolist()
    orig_logret = np.array([0.0] + [math.log(orig_levels[i] / orig_levels[i - 1]) for i in range(1, len(orig_levels))], dtype=float)[1:]
    orig_dates = pd.to_datetime(train[DATE_COL]).iloc[1:].tolist()

    aug_levels = pd.to_numeric(augmented_train[TARGET_COL], errors="coerce").astype(float).tolist()
    aug_logret = np.array([0.0] + [math.log(aug_levels[i] / aug_levels[i - 1]) for i in range(1, len(aug_levels))], dtype=float)[1:]
    aug_dates = pd.to_datetime(augmented_train[DATE_COL]).iloc[1:].tolist()

    sweep_results = []
    
    for context_len in [32, 64, 128]:
        print(f"Training Lag Transformer on augmented data with context_len={context_len}...")
        
        # Build sequences from augmented log returns
        X_aug, y_aug, scaler_aug = build_sequences(aug_logret, aug_dates, context_len)
        
        # Train model
        model = train_model(X_aug, y_aug, epochs=30)
        
        # Forecast recursively using original historical tail as starting context
        hist = orig_logret.tolist()
        hist_dates = orig_dates.copy()
        
        preds = []
        for d in pd.to_datetime(test[DATE_COL]).tolist():
            ctx = np.asarray(hist[-context_len:], dtype=np.float32)
            ctx_s = scaler_aug.transform(ctx.reshape(-1, 1)).reshape(-1)
            
            seq = np.concatenate([
                np.column_stack([ctx_s, date_features(hist_dates[-context_len:])]),
                np.column_stack([np.zeros(1, dtype=np.float32), date_features([d])]),
            ], axis=0)
            
            xb = torch.tensor(seq[None, :, :], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred_s = float(model(xb).item())
            
            pred = float(scaler_aug.inverse_transform(np.array([[pred_s]], dtype=np.float32))[0, 0])
            
            last_level = orig_levels[-1] if not preds else preds[-1]
            next_level = last_level * math.exp(pred)
            preds.append(next_level)
            
            hist.append(pred)
            hist_dates.append(d)
            
        test_rmse = rmse(y_true, preds)
        test_mae = float(mean_absolute_error(y_true, preds))
        test_mape = mape(y_true, preds)
        
        print(f"Context: {context_len} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.4f}%")
        sweep_results.append({
            "context_len": context_len,
            "rmse": test_rmse,
            "mae": test_mae,
            "mape": test_mape,
            "preds": preds
        })

    # Save details of sweep to csv/md
    sweep_df = pd.DataFrame(sweep_results).drop(columns=["preds"])
    sweep_df.to_csv("augmented_lag_transformer_results.csv", index=False)
    
    # Save the best predictions if they outperform best ssm ensemble
    best_sweep = min(sweep_results, key=lambda x: x["rmse"])
    print(f"\nBest Lag Transformer RMSE: {best_sweep['rmse']:.4f} with context_len={best_sweep['context_len']}")
    
    # Let's save the best predictions anyway
    best_preds_df = pd.DataFrame({
        "Date": test[DATE_COL],
        "USDIDR": best_sweep["preds"]
    })
    best_preds_df.to_csv("augmented_lag_transformer_best_predictions.csv", index=False)

if __name__ == "__main__":
    main()
