#!/usr/bin/env python3
"""GRU forecasting model for daily USDIDR time series.

This script uses a Gated Recurrent Unit (GRU) with Optuna hyperparameter optimization.
Includes critical target-difference features and aggressive hyperparameter search.

Requirements
------------
torch, pandas, numpy, scikit-learn, optuna

Example
-------
python gru_forecast.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "USDIDR"
LOOKBACK = 63  # Increased to 63 days (Quarterly view)
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 7200  # 2 hours max search time

# Training Defaults
BATCH_SIZE = 64
EPOCHS = 150  # More epochs for deeper models
EARLY_STOPPING_PATIENCE = 20

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Infer a date column if one exists."""
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates:
        return candidates[0]
    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            return first_col
    except Exception:
        pass
    return None


def sort_by_date_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    date_col = infer_date_col(df)
    out = df.copy()
    if date_col is not None:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def safe_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.where(s > 0))


def one_step_diff(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").diff()


def cyclical_encoding(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    theta = 2.0 * np.pi * values / period
    return np.sin(theta), np.cos(theta)


def get_existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# -----------------------------
# Exogenous feature engineering
# -----------------------------

def engineer_exogenous_features(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Create features from known-at-prediction-time exogenous variables."""
    out = df.copy()
    for c in out.columns:
        if c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["time_index"] = np.arange(len(out), dtype=float)

    # Calendar features
    if date_col is not None:
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out["dow"] = dt.dt.dayofweek.astype(float)
        out["month"] = dt.dt.month.astype(float)
        out["doy"] = dt.dt.dayofyear.astype(float)
        dow_sin, dow_cos = cyclical_encoding(out["dow"], 7)
        doy_sin, doy_cos = cyclical_encoding(out["doy"], 365.25)
        out["dow_sin"] = dow_sin
        out["dow_cos"] = dow_cos
        out["doy_sin"] = doy_sin
        out["doy_cos"] = doy_cos

    # Spread
    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]

    # Logs
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log(out[c])

    # Differences
    diff_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for c in get_existing_cols(out, diff_cols):
        out[f"diff_{c}"] = one_step_diff(out[c])

    # Log Returns
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"logret_{c}"] = np.log(out[c] / out[c].shift(1))

    # Rolling Volatility
    for c in get_existing_cols(out, diff_cols):
        diff_col = f"diff_{c}"
        if diff_col in out.columns:
            out[f"vol_{c}_5"] = out[diff_col].rolling(window=5, min_periods=1).std()
            out[f"vol_{c}_21"] = out[diff_col].rolling(window=21, min_periods=1).std()

    # Fill NaNs
    out = out.fillna(method='bfill').fillna(0)
    return out


# -----------------------------
# Sequence Dataset Construction
# -----------------------------

def prepare_sequences(
    train_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
    val_split: float = 0.2,
    date_col: Optional[str] = None,

) -> Tuple:
    """
    Prepares 3D sequences. 
    CRITICAL: Adds explicit lagged differences of the TARGET to the input features.
    """
    data = exog_df.copy()
    data[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    
    # --- NEW: Add Target Momentum Features ---
    # Calculate log returns of the target
    data[f"logret_{target_col}"] = np.log(data[target_col]).diff()
    
    # Create lags for target returns (Momentum indicators)
    for lag in [1, 5, 21]:
        data[f"logret_{target_col}_lag{lag}"] = data[f"logret_{target_col}"].shift(lag)
    
    # Create lagged target level (The anchor)
    data[f"{target_col}_lag1"] = data[target_col].shift(1)
    
    # --- Feature Selection ---
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Define final feature list (exclude raw target, include engineered features)
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    # Clean NaNs resulting from shifting/diffing
    # We drop rows where ANY of our key features are NaN
    clean_data = data.dropna(subset=[f"{target_col}_lag1"] + feature_cols)
    
    # Calculate Deltas (Target for prediction)
    # We predict the change from t-1 to t
    # y_raw_deltas = np.log(clean_data[target_col].values) - np.log(clean_data[f"{target_col}_lag1"].values)
    y_raw_levels = clean_data[target_col].values
    X_raw = clean_data[feature_cols].values
    
    # Split Train/Val (Time-series aware)
    date_col_vals = clean_data.index if date_col is None else pd.to_datetime(clean_data[date_col])
    split_idx = (date_col_vals < "2021-01-01").sum()
    
    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_val_raw = y_raw_levels[:split_idx], y_raw_levels[split_idx:]
    
    # Fit RobustScalers on Training Data Only
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_val_scaled = scaler_X.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    
    # Create Sequences Helper
    def make_seqs(X, y, lb):
        Xs, ys = [], []
        for i in range(lb, len(X)):
            Xs.append(X[i-lb:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = make_seqs(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = make_seqs(X_val_scaled, y_val_scaled, lookback)
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, scaler_X, scaler_y, feature_cols


# -----------------------------
# PyTorch Model
# -----------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, dropout: float):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


# -----------------------------
# Training Logic
# -----------------------------

def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            # Gradient clipping to prevent exploding gradients in deep RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_state:
        model.load_state_dict(best_state)
        
    return best_val_loss


# -----------------------------
# Recursive forecasting
# -----------------------------

def forecast_test_period(model, train_df, test_df, combined_exog,
                          scaler_X, scaler_y, feature_cols,
                          target_col=TARGET_COL, lookback=LOOKBACK):
    model.eval()
    preds = []

    full_exog = engineer_exogenous_features(combined_exog, infer_date_col(combined_exog))
    
    # Build feature matrix for test using ONLY known exog (no recursive target)
    # Last known USDIDR = last value in train
    last_known_level = pd.to_numeric(train_df[target_col], errors="coerce").iloc[-1]
    last_known_return = np.log(last_known_level / pd.to_numeric(train_df[target_col], errors="coerce").iloc[-2])

    test_start = len(train_df)
    
    for i in range(len(test_df)):
        t = test_start + i
        window_start = t - lookback
        
        # Build window dari combined exog — TIDAK pakai predicted USDIDR
        window = full_exog.iloc[window_start:t].copy()
        
        # Inject last KNOWN lag (tidak diupdate tiap step)
        if f"{target_col}_lag1" in feature_cols:
            window[f"{target_col}_lag1"] = last_known_level  # anchor ke last real value

        # Zero-fill target-derived cols that cannot be computed in test period
        # (logret_USDIDR, logret_USDIDR_lag*, etc. — only available during training)
        for col in feature_cols:
            if col not in window.columns:
                window[col] = 0.0

        row_vals = window[feature_cols].fillna(0).values
        row_scaled = scaler_X.transform(row_vals)
        
        X_tensor = torch.tensor(row_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()[0, 0]
        
        pred_level = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(pred_level)

    return np.array(preds, dtype=float)
# -----------------------------
# Main runner
# -----------------------------

def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    submission_csv: Path,
    actual_test_csv: Optional[Path] = None,
) -> None:
    set_seed(RANDOM_STATE)
    
    # Load Data
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_test_raw = pd.read_csv(actual_test_csv) if actual_test_csv else None

    train_raw = sort_by_date_if_possible(train_raw)
    test_raw = sort_by_date_if_possible(test_raw)
    if actual_test_raw is not None:
        actual_test_raw = sort_by_date_if_possible(actual_test_raw)

    train_date_col = infer_date_col(train_raw)
    train_exog = train_raw.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_raw.copy()

    # Feature Engineering
    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    combined_exog = engineer_exogenous_features(combined_exog, date_col=train_date_col)
    
    # Prepare Sequences
    X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_cols = prepare_sequences(
        train_df=train_raw,
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True),
        target_col=TARGET_COL,
        lookback=LOOKBACK,
        val_split=0.2,
        date_col=train_date_col,
    )
    
    print(f"Data Prepared: Train={len(X_train)}, Val={len(X_val)}, Features={len(feature_cols)}")

    # -----------------------------
    # Optuna Objective (Aggressive)
    # -----------------------------
    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "dropout": trial.suggest_float("dropout", 0.3, 0.6),  # lebih agresif,
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64])
        }
        
        model = GRUModel(
            input_size=len(feature_cols),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(DEVICE)
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        val_loss = train_model(
            model, train_loader, val_loader, 
            epochs=50, # Faster trials
            lr=params['learning_rate'], 
            patience=10,
            device=DEVICE
        )
        
        return val_loss

    # Run Optimization
    print("Starting Aggressive Optuna Search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)
    
    print("\nBest Trial:")
    print(f"  Value (Loss): {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")
    
    best_params = study.best_params

    # Final Training
    X_full = np.concatenate((X_train, X_val), axis=0)
    y_full = np.concatenate((y_train, y_val), axis=0)
    
    full_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_full, dtype=torch.float32),
        torch.tensor(y_full, dtype=torch.float32).unsqueeze(1)
    )
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    final_model = GRUModel(
        input_size=len(feature_cols),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    
    print("Training final model on full dataset...")
    train_model(
        final_model, full_loader, val_loader,
        epochs=EPOCHS,
        lr=best_params['learning_rate'],
        patience=EARLY_STOPPING_PATIENCE,
        device=DEVICE
    )

    # Forecast
    test_preds = forecast_test_period(
        model=final_model,
        train_df=train_raw,
        test_df=test_raw,
        combined_exog=combined_exog,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        lookback=LOOKBACK,
    )

    # Save
    submission = pd.DataFrame({TARGET_COL: test_preds})
    test_date_col = infer_date_col(test_raw)
    if test_date_col and test_date_col in test_raw.columns:
        submission.insert(0, "Date", test_raw[test_date_col].values)
    
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Saved submission to: {submission_csv}")

    # Benchmark
    if actual_test_raw is not None:
        if TARGET_COL not in actual_test_raw.columns:
            raise ValueError(f"actual_test_csv must contain target column '{TARGET_COL}'.")
        actual = pd.to_numeric(actual_test_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        min_len = min(len(actual), len(test_preds))
        rmse = np.sqrt(mean_squared_error(actual[:min_len], test_preds[:min_len]))
        print(f"True test RMSE: {rmse:.4f}")

        naive_rmse = np.sqrt(mean_squared_error(
            actual[:min_len][1:], 
            actual[:min_len][:-1]  # naive = predict yesterday's value
        ))
        print(f"Naive baseline RMSE (yesterday = today): {naive_rmse:.4f}")
        print(f"Your model RMSE: {rmse:.4f}")
        print(f"Beats naive: {rmse < naive_rmse}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USDIDR GRU forecasting with Optuna")
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