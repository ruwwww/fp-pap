#!/usr/bin/env python3
"""SOTA Hybrid Pipeline: ElasticNet (Macro Trend) + PatchTST (Micro Volatility).

Architecture Strategy:
  1. ElasticNet: Belajar memprediksi log-return untuk membangun garis tren linear dasar (Base Trend).
  2. Residual Extractor: Menghitung selisih (error) antara aktual dan tren ElasticNet di data train.
  3. PatchTST: Dilatih khusus pada data Residual (stasioner berpusat di 0) secara Univariat & Direct Forecasting.
  4. Final Inference: Prediksi Tren ElasticNet + Prediksi Residual PatchTST.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from transformers import PatchTSTConfig, PatchTSTForPrediction

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Configuration & Hyperparameters
# ──────────────────────────────────────────────
TARGET_COL = "USDIDR"
RANDOM_STATE = 42
MAX_DAILY_LOG_RETURN = 0.03

# PatchTST Hyperparameters (Tuned for Residuals)
LOOKBACK_WINDOW = 378
PATCH_LEN = 24
STRIDE = 10
D_MODEL = 256
N_HEADS = 2
BATCH_SIZE = 32
EPOCHS = 30           # Dinaikkan agar model matang dan tidak telat fase (delay)
LEARNING_RATE = 3e-4  # Diturunkan untuk akurasi konvergensi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 2. Utilities & Feature Engineering (Strict Causal)
# ──────────────────────────────────────────────
def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates: return candidates[0]
    return None

def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan anomali ekstrem dan bad data menggunakan interpolasi."""
    out = df.copy()
    
    if TARGET_COL in out.columns:
        # 1. Hapus nilai yang secara historis mustahil (Misal: Kurs USD/IDR di bawah Rp 6.000)
        # Menghadapi data 0, angka ratusan, atau missing value terselubung
        out.loc[out[TARGET_COL] < 6000, TARGET_COL] = np.nan
        
        # 2. Deteksi Anomali Harian Ekstrem (Z-Score / Pct Change)
        # Jika nilai kurs mendadak berubah lebih dari 10% dalam sehari, itu 99% glitch data
        pct_change = out[TARGET_COL].pct_change().abs()
        out.loc[pct_change > 0.10, TARGET_COL] = np.nan
        
        # 3. Interpolasi Linear
        # Menyambung garis putus (NaN) dengan menarik garis lurus dari titik sebelum & sesudahnya
        out[TARGET_COL] = out[TARGET_COL].interpolate(method='linear')
        
        # 4. Fallback jika anomali ada di baris paling pertama atau terakhir
        out[TARGET_COL] = out[TARGET_COL].bfill().ffill()

    # Bersihkan kolom eksogen lain (jika ada)
    if "OIL" in out.columns:
        out["OIL"] = out["OIL"].clip(lower=0)
    if "VIX" in out.columns:
        cap = out["VIX"].quantile(0.99)
        out["VIX"] = out["VIX"].clip(upper=cap)
        
    return out

def engineer_exog(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Membangun fitur makro kausal (t-1)."""
    out = df.copy()
    raw_exog = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    
    # Lag semua eksogen ke t-1
    for c in raw_exog:
        if c in out.columns:
            out[f"{c}_lag1"] = out[c].shift(1)
            out[f"{c}_diff_lag1"] = out[c].diff().shift(1)
            
    # Drop raw t-0 untuk mencegah kebocoran masa depan
    out = out.drop(columns=[c for c in raw_exog if c in out.columns], errors="ignore")
    return out

def target_context(levels_history: List[float]) -> Dict[str, float]:
    """Lag historis target USDIDR murni hingga t-1."""
    hist = pd.Series(levels_history, dtype="float64")
    feats = {}
    feats["usdidr_lag1"] = hist.iloc[-1] if len(hist) >= 1 else np.nan
    feats["usdidr_lag5"] = hist.iloc[-5] if len(hist) >= 5 else np.nan
    if len(hist) >= 2:
        feats["usdidr_diff_1"] = hist.diff().iloc[-1]
    return feats


# ──────────────────────────────────────────────
# 3. PyTorch Dataset untuk PatchTST
# ──────────────────────────────────────────────
class ResidualDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        past_v = self.data[idx : idx + self.lookback]
        future_v = self.data[idx + self.lookback : idx + self.lookback + self.horizon]
        return past_v, future_v


# ──────────────────────────────────────────────
# 4. Main Pipeline Runner
# ──────────────────────────────────────────────
def run_pipeline(train_csv: Path, test_csv: Path, submission_csv: Path, actual_test_csv: Optional[Path] = None):
    print(f"[{DEVICE.type.upper()}] Memulai Pipeline Hybrid ElasticNet + PatchTST...")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # --- A. DATA LOADING & PREP ---
    train_raw = fix_anomalies(pd.read_csv(train_csv))
    test_raw = fix_anomalies(pd.read_csv(test_csv))
    actual_raw = fix_anomalies(pd.read_csv(actual_test_csv)) if actual_test_csv else None

    date_col = infer_date_col(train_raw)
    train_raw = train_raw.sort_values(date_col).reset_index(drop=True)
    test_raw = test_raw.sort_values(date_col).reset_index(drop=True)
    
    horizon = len(test_raw)
    
    combined_exog = pd.concat([train_raw.drop(columns=[TARGET_COL, date_col], errors="ignore"), 
                               test_raw.drop(columns=[TARGET_COL, date_col], errors="ignore")], ignore_index=True)
    combined_exog = engineer_exog(combined_exog, date_col=None)

    # --- B. TRAINING ELASTICNET (BASE TREND) ---
    print("\n[1/4] Melatih ElasticNet untuk Tren Dasar...")
    train_levels = pd.to_numeric(train_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    exog_cols = list(combined_exog.columns)
    
    rows_X, y_logret = [], []
    min_hist = 6
    for t in range(min_hist, len(train_raw)):
        ctx = target_context(train_levels[:t])
        row = {c: combined_exog.iloc[t][c] for c in exog_cols if c in combined_exog.columns}
        row.update(ctx)
        rows_X.append(row)
        y_logret.append(np.log(train_levels[t]) - np.log(train_levels[t-1]))

    X_train_df = pd.DataFrame(rows_X)
    y_train_series = pd.Series(y_logret)
    feature_names = X_train_df.columns.tolist()

    en_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=10000, random_state=RANDOM_STATE))
    ])
    en_model.fit(X_train_df, y_train_series)

    # --- C. CALCULATE RESIDUALS IN-SAMPLE ---
    print("[2/4] Mengekstrak Residu Stasioner...")
    fitted_logrets_clipped = np.clip(en_model.predict(X_train_df), -MAX_DAILY_LOG_RETURN, MAX_DAILY_LOG_RETURN)
    
    aligned_prev_levels = train_levels[min_hist-1 : -1]
    aligned_actual_levels = train_levels[min_hist:]
    fitted_levels = aligned_prev_levels * np.exp(fitted_logrets_clipped)
    
    # Residu absolut: Harga Asli - Harga Tebakan ElasticNet
    train_residuals = aligned_actual_levels - fitted_levels

    # --- D. TRAINING PATCHTST ON RESIDUALS ---
    print(f"\n[3/4] Melatih PatchTST secara Univariat pada Data Residual...")
    res_scaler = RobustScaler()
    train_res_scaled = res_scaler.fit_transform(train_residuals.reshape(-1, 1))

    res_dataset = ResidualDataset(train_res_scaled, LOOKBACK_WINDOW, horizon)
    res_loader = DataLoader(res_dataset, batch_size=BATCH_SIZE, shuffle=True)

    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=LOOKBACK_WINDOW,
        prediction_length=horizon,
        patch_length=PATCH_LEN,
        stride=STRIDE,
        d_model=D_MODEL,
        num_attention_heads=N_HEADS,
        encoder_layers=3,
        scaling="std",
        dropout=0.2,
    )
    patchtst = PatchTSTForPrediction(config).to(DEVICE)
    
    optimizer = torch.optim.AdamW(patchtst.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0) # Huber Loss agar tahan banting

    patchtst.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for past_v, future_v in res_loader:
            past_v, future_v = past_v.to(DEVICE), future_v.to(DEVICE)
            optimizer.zero_grad()
            outputs = patchtst(past_values=past_v, future_values=future_v)
            loss = criterion(outputs.prediction_outputs, future_v)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * past_v.size(0)
            
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f" Epoch [{epoch:02d}/{EPOCHS:02d}] - Residual Loss: {total_loss / len(res_dataset):.6f}")

    # --- E. FORECASTING (HYBRID INFERENCE) ---
    print("\n[4/4] Melakukan Prediksi Hybrid pada Test Set...")
    
    # 1. Base Trend Forecast (Recursive ElasticNet)
    test_trend_preds = []
    history_levels = list(train_levels)
    start_idx = len(train_raw)
    
    for i in range(horizon):
        idx = start_idx + i
        ctx = target_context(history_levels)
        row = {c: combined_exog.iloc[idx][c] for c in exog_cols if c in combined_exog.columns}
        row.update(ctx)
        
        X_row = pd.DataFrame([row]).reindex(columns=feature_names, fill_value=np.nan)
        delta_en = np.clip(float(en_model.predict(X_row)[0]), -MAX_DAILY_LOG_RETURN, MAX_DAILY_LOG_RETURN)
        
        next_level = history_levels[-1] * np.exp(delta_en)
        test_trend_preds.append(next_level)
        history_levels.append(next_level)
        
    test_trend_preds = np.array(test_trend_preds)

    # 2. Residual Forecast (Direct PatchTST)
    patchtst.eval()
    past_res_context = train_res_scaled[-LOOKBACK_WINDOW:]
    past_res_tensor = torch.tensor(past_res_context, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_outputs = patchtst(past_values=past_res_tensor).prediction_outputs
        test_res_scaled = pred_outputs.squeeze(0).cpu().numpy()
        
    test_res_unscaled = res_scaler.inverse_transform(test_res_scaled)[:, 0]

    # 3. Final Fusion
    final_hybrid_preds = test_trend_preds + test_res_unscaled

    # --- F. SAVE & EVALUATE ---
    submission = pd.DataFrame({TARGET_COL: final_hybrid_preds})
    submission.insert(0, "Date", test_raw[date_col].values)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"\nDisimpan ke: {submission_csv}")

    if actual_raw is not None:
        actual_values = pd.to_numeric(actual_raw[TARGET_COL], errors="coerce").to_numpy()
        rmse_hybrid = float(np.sqrt(mean_squared_error(actual_values, final_hybrid_preds)))
        rmse_naive = float(np.sqrt(mean_squared_error(actual_values[1:], actual_values[:-1])))
        rmse_trend_only = float(np.sqrt(mean_squared_error(actual_values, test_trend_preds)))

        print(f"\n{'='*55}")
        print(f" TRUE TEST BENCHMARK EVALUATION (HYBRID PIPELINE)")
        print(f"{'='*55}")
        print(f" Naive Baseline RMSE       : {rmse_naive:.4f}")
        print(f" ElasticNet Only (Trend)   : {rmse_trend_only:.4f}")
        print(f" Hybrid PatchTST SOTA RMSE : {rmse_hybrid:.4f}")
        print(f"{'='*55}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=Path, required=True)
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--submission_csv", type=Path, required=True)
    p.add_argument("--actual_test_csv", type=Path, default=None)
    args = p.parse_args()
    
    run_pipeline(args.train_csv, args.test_csv, args.submission_csv, args.actual_test_csv)