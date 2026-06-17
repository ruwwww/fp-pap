#!/usr/bin/env python3
"""GRU forecasting model for daily USDIDR time series.

This script adapts the ElasticNet template to use a Gated Recurrent Unit (GRU)
Deep Learning model via PyTorch. It now includes a manual hyperparameter search.

Requirements
------------
torch, pandas, numpy, scikit-learn

Example
-------
python gru_forecast.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Configuration & Hyperparameter Grid
# -----------------------------
TARGET_COL = "USDIDR"
DEFAULT_LAGS = [1, 5, 21, 63] 
LOOKBACK = 21  # Number of past days to feed into the GRU
BATCH_SIZE = 64
EPOCHS = 50 # Reduced epochs for faster search
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42

# Hyperparameter Search Grid
HP_GRID = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.3],
    "learning_rate": [0.001, 0.0005]
}

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

    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]

    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log(out[c])

    diff_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for c in get_existing_cols(out, diff_cols):
        out[f"diff_{c}"] = one_step_diff(out[c])

    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"logret_{c}"] = np.log(out[c] / out[c].shift(1))

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
) -> Tuple:
    """
    Prepares 3D sequences for the GRU and splits into Train/Val.
    """
    data = exog_df.copy()
    data[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    
    # Prepare Data
    data_lagged_target = data.copy()
    data_lagged_target[f"{target_col}_lag1"] = data[target_col].shift(1)
    
    # Identify numeric columns
    # Use select_dtypes to ensure we don't pick up datetime columns like 'Date'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols_raw = [c for c in numeric_cols if c != target_col]
    final_feature_cols = [c for c in feature_cols_raw if c != f"{target_col}_lag1"] + [f"{target_col}_lag1"]
    
    # Clean NaNs
    clean_data = data_lagged_target.dropna(subset=[f"{target_col}_lag1"] + final_feature_cols)
    
    # Calculate Deltas
    y_raw_deltas = np.log(clean_data[target_col].values) - np.log(clean_data[f"{target_col}_lag1"].values)
    X_raw = clean_data[final_feature_cols].values
    
    # Split into Train/Validation BEFORE scaling to prevent leakage
    split_idx = int(len(X_raw) * (1.0 - val_split))
    
    X_train_raw, X_val_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_val_raw = y_raw_deltas[:split_idx], y_raw_deltas[split_idx:]
    
    # Fit Scalers on Training Data Only
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    
    X_val_scaled = scaler_X.transform(X_val_raw) # Transform val with train stats
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1)).flatten()
    
    # Create Sequences
    def make_seqs(X, y, lb):
        Xs, ys = [], []
        for i in range(lb, len(X)):
            Xs.append(X[i-lb:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = make_seqs(X_train_scaled, y_train_scaled, lookback)
    X_val_seq, y_val_seq = make_seqs(X_val_scaled, y_val_scaled, lookback)
    
    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, scaler_X, scaler_y, final_feature_cols


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
# Training & Evaluation Logic
# -----------------------------

def train_model(model, train_loader, val_loader, epochs, lr, patience):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
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

def forecast_test_period(
    model: nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    combined_exog: pd.DataFrame,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
    lookback: int = LOOKBACK,
) -> np.ndarray:
    model.eval()
    preds = []
    
    full_exog = engineer_exogenous_features(combined_exog, infer_date_col(combined_exog))
    lag_col = f"{target_col}_lag1"
    
    n_total = len(full_exog)
    levels_known = [np.nan] * (n_total + 1)
    for i in range(len(train_df)):
        levels_known[i] = pd.to_numeric(train_df[target_col], errors="coerce").iloc[i]
        
    def get_scaled_row(t: int, current_levels: List[float]) -> np.ndarray:
        exog_row = full_exog.iloc[t]
        row_dict = {}
        for c in feature_cols:
            if c == lag_col:
                row_dict[c] = current_levels[t-1]
            else:
                row_dict[c] = exog_row.get(c, 0.0)
        
        row_df = pd.DataFrame([row_dict], columns=feature_cols)
        return scaler_X.transform(row_df.to_numpy())[0]

    train_end_idx = len(train_df) - 1
    test_start_idx = len(train_df)
    
    current_window = []
    start_window_idx = train_end_idx - lookback + 1
    
    # Initialize window with tail of training data
    for i in range(start_window_idx, train_end_idx + 1):
        row_scaled = get_scaled_row(i, levels_known)
        current_window.append(row_scaled)
        
    for i in range(len(test_df)):
        t = test_start_idx + i
        
        # Get input for time t
        input_row_t = get_scaled_row(t, levels_known)
        current_window.append(input_row_t)
        if len(current_window) > lookback:
            current_window.pop(0)
            
        X_tensor = torch.tensor([current_window], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_scaled_delta = model(X_tensor)
            pred_scaled_delta = pred_scaled_delta.cpu().numpy()[0, 0]
            
        pred_raw_delta = scaler_y.inverse_transform([[pred_scaled_delta]])[0, 0]
        level_t = levels_known[t-1] * np.exp(pred_raw_delta)
        preds.append(level_t)
        levels_known[t] = level_t
        
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
    
    # Load data
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

    # Prepare Features and Sequences
    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    effective_date_col = train_date_col
    combined_exog = engineer_exogenous_features(combined_exog, date_col=effective_date_col)
    
    # Prepare train/val splits
    X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_cols = prepare_sequences(
        train_df=train_raw,
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True),
        target_col=TARGET_COL,
        lookback=LOOKBACK,
        val_split=0.2
    )
    
    print(f"Training samples: {len(X_train)} | Validation samples: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")
    
    # Create Loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    # -----------------------------
    # Hyperparameter Search Loop
    # -----------------------------
    keys, values = zip(*HP_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    best_score = float('inf')
    best_params = None
    best_model_state = None
    
    for params in combinations:
        print(f"Testing params: {params}")
        
        # Init Model
        model = GRUModel(
            input_size=len(feature_cols),
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(DEVICE)
        
        # Loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train
        val_loss = train_model(
            model, train_loader, val_loader, 
            epochs=EPOCHS, 
            lr=params['learning_rate'], 
            patience=EARLY_STOPPING_PATIENCE
        )
        
        print(f"Val Loss: {val_loss:.6f}")
        
        if val_loss < best_score:
            best_score = val_loss
            best_params = params
            best_model_state = model.state_dict()
            print(">>> New best model found!")

    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best Validation Loss: {best_score:.6f}")
    
    # Load Best Model
    final_model = GRUModel(
        input_size=len(feature_cols),
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(DEVICE)
    final_model.load_state_dict(best_model_state)

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

    # Save Submission
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
        actual = actual[:min_len]
        test_preds = test_preds[:min_len]
        
        # Fix for sklearn 1.4+
        rmse = np.sqrt(mean_squared_error(actual, test_preds))
        print(f"True test RMSE: {rmse:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USDIDR GRU forecasting template")
    p.add_argument("--train_csv", type=Path, required=True, help="Path to training CSV")
    p.add_argument("--test_csv", type=Path, required=True, help="Path to test CSV")
    p.add_argument("--submission_csv", type=Path, required=True, help="Output submission CSV")
    p.add_argument(
        "--actual_test_csv",
        type=Path,
        default=None,
        help="Optional CSV with actual test USDIDR for honest benchmarking",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        submission_csv=args.submission_csv,
        actual_test_csv=args.actual_test_csv,
    )