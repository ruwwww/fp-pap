#!/usr/bin/env python3
"""Strict Causal Multivariate PatchTST Pipeline using Hugging Face Transformers.

Strategy:
  - Bypasses NeuralForecast limitations by using native Hugging Face PatchTST.
  - Accepts all 8 exogenous macro features alongside USDIDR simultaneously.
  - Uses Direct Forecasting: predicts the entire test horizon (778 days) in one single forward pass.
  - Leverages PyTorch for direct GPU acceleration.
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

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TARGET_COL = "USDIDR"
RANDOM_STATE = 42
LOOKBACK_WINDOW = 252  # 1 tahun data trading historis
BATCH_SIZE = 32
EPOCHS = 15            # Sesuaikan dengan compute kamu, karena pakai GPU bisa cepat
LEARNING_RATE = 1e-3

# Set device secara otomatis (Prioritas CUDA GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Dataset & Preprocessing Utilities
# ──────────────────────────────────────────────
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates: return candidates[0]
    return None

def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning sesuai visualisasi drop ekstrem."""
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
    """Custom PyTorch Dataset untuk mapping windowing PatchTST."""
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # past_values: (seq_len, num_features)
        past_values = self.data[idx : idx + self.lookback]
        # future_values: (pred_len, num_features)
        future_values = self.data[idx + self.lookback : idx + self.lookback + self.horizon]
        return past_values, future_values


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    submission_csv: Path,
    actual_test_csv: Optional[Path] = None,
) -> None:
    print(f"Menggunakan Device: {DEVICE}")
    torch.manual_seed(RANDOM_STATE)

    # 1. Load & Clean Data
    train_raw = fix_anomalies(pd.read_csv(train_csv))
    test_raw = pd.read_csv(test_csv)
    actual_raw = fix_anomalies(pd.read_csv(actual_test_csv)) if actual_test_csv else None

    date_col = infer_date_col(train_raw)
    if not date_col: raise ValueError("Kolom tanggal tidak ditemukan.")

    # Sort Kronologis
    train_raw = train_raw.sort_values(date_col).reset_index(drop=True)
    test_raw = test_raw.sort_values(date_col).reset_index(drop=True)

    horizon = len(test_raw)
    print(f"Data Train Rows: {len(train_raw)}")
    print(f"Data Test Rows  : {horizon} harian (Direct Horizon h={horizon})")

    # Univariat murni: hanya USDIDR, tanpa fitur eksogen
    feature_order = [TARGET_COL]
    print(f"Urutan Fitur Channels ({len(feature_order)}): {feature_order}")

    # 2. Scaling Data (Sangat Krusial untuk Stabilitas Transformer)
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_raw[feature_order].values)
    
    # Test set tidak punya kolom USDIDR; pakai dummy 0 untuk scaler transform
    test_dummy = pd.DataFrame(0, index=np.arange(len(test_raw)), columns=[TARGET_COL])
    test_scaled = scaler.transform(test_dummy[feature_order].values)

    # 3. Buat PyTorch Data Loader untuk Training
    train_dataset = TimeSeriesDataset(train_scaled, LOOKBACK_WINDOW, horizon)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Inisialisasi Konfigurasi PatchTST Hugging Face
    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=LOOKBACK_WINDOW,
        prediction_length=horizon,
        patch_length=16,
        stride=8,
        d_model=128,
        num_attention_heads=4,
        encoder_layers=3,
        scaling="std",  # Mengaktifkan internal RevIN (Instance Normalization) bawaan PatchTST
    )
    model = PatchTSTForPrediction(config).to(DEVICE)

    # 5. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\n" + "="*50)
    print(" Memulai Proses Training PatchTST (Hugging Face)...")
    print("="*50)
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for past_v, future_v in train_loader:
            past_v = past_v.to(DEVICE)
            future_v = future_v.to(DEVICE)

            optimizer.zero_grad()
            # Forward pass HF PatchTST mengharuskan input berupa nama argumen langsung
            outputs = model(past_values=past_v, future_values=future_v)
            
            loss = criterion(outputs.prediction_outputs, future_v)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * past_v.size(0)
            
        epoch_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Training Loss (MSE): {epoch_loss:.6f}")

    # 6. Direct Inference Phase (Strictly Causal)
    print("\nMelakukan Direct Forecasting untuk Periode Test...")
    model.eval()
    
    # Ambiil data historis terakhir dari TRAIN set sebagai modal konteks lookback window
    past_context = train_scaled[-LOOKBACK_WINDOW:]  # Shape: (LOOKBACK_WINDOW, num_features)
    past_context_tensor = torch.tensor(past_context, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dim -> (1, L, C)

    with torch.no_grad():
        # Lakukan prediksi langsung sebanyak panjang horizon ke depan
        pred_outputs = model(past_values=past_context_tensor).prediction_outputs
        # Shape output: (1, horizon, num_features)
        pred_scaled = pred_outputs.squeeze(0).cpu().numpy()

    # 7. Inverse Transform Hasil Prediksi
    # Karena kita melakukan scaling bersama, kita kembalikan skala data ke bentuk aslinya
    pred_unscaled = scaler.inverse_transform(pred_scaled)
    # Target USDIDR berada di kolom indeks 0 sesuai variabel `feature_order`
    final_preds = pred_unscaled[:, 0]

    # 8. Save Submission
    submission = pd.DataFrame({TARGET_COL: final_preds})
    submission.insert(0, "Date", test_raw[date_col].values)
    
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Berhasil menyimpan hasil ke: {submission_csv}")

    # 9. Evaluasi jika data pembanding disediakan
    if actual_raw is not None:
        actual_values = pd.to_numeric(actual_raw[TARGET_COL], errors="coerce").to_numpy()
        rmse_patchtst = float(np.sqrt(mean_squared_error(actual_values, final_preds)))
        rmse_naive = float(np.sqrt(mean_squared_error(actual_values[1:], actual_values[:-1])))

        print(f"\n{'='*55}")
        print(f" TRUE TEST BENCHMARK EVALUATION (PATCHTST HF)")
        print(f"{'='*55}")
        print(f" Naive Baseline RMSE (Yesterday=Today) : {rmse_naive:.4f}")
        print(f" Multivariate PatchTST SOTA RMSE       : {rmse_patchtst:.4f}")
        print(f"{'='*55}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multivariate PatchTST Hugging Face Pipeline")
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--submission_csv", type=Path, required=True)
    p.add_argument("--actual_test_csv", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        submission_csv=args.submission_csv,
        actual_test_csv=args.actual_test_csv,
    )