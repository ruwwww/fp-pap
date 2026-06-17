#!/usr/bin/env python3
"""GRU forecasting model for daily USDIDR time series.

This script adapts the ElasticNet template to use a Gated Recurrent Unit (GRU)
Deep Learning model via PyTorch. It maintains the core logic:
- forecast USDIDR *changes* (deltas) rather than raw levels
- use a sequence of past USDIDR levels and exogenous features as input
- add macro feature transforms (logs, differences, spreads)
- preserve time order
- recursively forecast the test period one day at a time

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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "USDIDR"
DEFAULT_LAGS = [1, 5, 21, 63]  # Note: GRU uses a sequence window instead of explicit lag cols
LOOKBACK = 21  # Number of past days to feed into the GRU
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10
RANDOM_STATE = 42

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Infer a date column if one exists."""
    candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "ds", "timestamp"}]
    if candidates:
        return candidates[0]

    # Try the first column if it can be parsed as dates.
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
    """Log transform only on strictly positive values."""
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

    # Ensure numeric where possible.
    for c in out.columns:
        if c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Time index across the concatenated train+test horizon.
    out["time_index"] = np.arange(len(out), dtype=float)

    # Calendar features, if a date column exists.
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

    # Core spread feature.
    if "BI_rate" in out.columns and "US_rate" in out.columns:
        out["rate_spread"] = out["BI_rate"] - out["US_rate"]

    # Log transforms for long-run growers.
    for c in ["GOLD", "SP500", "IHSG"]:
        if c in out.columns:
            out[f"log_{c}"] = safe_log(out[c])

    # Simple first differences for exogenous series.
    diff_cols = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for c in get_existing_cols(out, diff_cols):
        out[f"diff_{c}"] = one_step_diff(out[c])

    # Log returns for positive macro series.
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
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, List[str]]:
    """
    Prepares 3D sequences for the GRU.
    
    Returns:
        X: (samples, lookback, features)
        y: (samples, 1) - log-difference of target
        scaler_X: fitted scaler for input features
        scaler_y: fitted scaler for target
        feature_names: list of feature columns in order
    """
    # 1. Merge Target with Exogenous Features
    # We need the target (USDIDR) as a feature input for the next time step
    data = exog_df.copy()
    data[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    
    # 2. Select only numeric columns for modeling
    # Drop datetime columns if any exist in the frame
    cols_to_keep = data.select_dtypes(include=[np.number]).columns.tolist()
    data = data[cols_to_keep]
    
    # Ensure target is first or last consistently? 
    # We just need to know the columns.
    feature_cols = [c for c in cols_to_keep if c != target_col]
    
    # Reorder: Target is important, so let's place it first or manage it explicitly.
    # The logic: Input[t] contains features at t. Target is level at t.
    # We want to predict Level[t] - Level[t-1].
    # For the sequence window, we include Level[t-1] as a feature.
    
    # Clean NaNs
    data = data.dropna()
    
    # 3. Calculate Target (Log Difference)
    # y[t] = log(Level[t]) - log(Level[t-1])
    levels = data[target_col].values
    deltas = np.log(levels[1:]) - np.log(levels[:-1])
    
    # 4. Scale Features (X)
    # We fit on the full available history (train + part of exog used for context)
    # But strictly, scaler should be fit on Train only to avoid leakage.
    # Here 'data' corresponds to the train_df length + some buffer if exog_df was larger, 
    # but typically we pass the train slice.
    
    scaler_X = StandardScaler()
    scaled_features = scaler_X.fit_transform(data[feature_cols + [target_col]])
    
    # 5. Scale Target (y)
    # Reshape y for scaler
    scaler_y = StandardScaler()
    scaled_deltas = scaler_y.fit_transform(deltas.reshape(-1, 1)).flatten()
    
    # 6. Construct Sequences
    X_seq = []
    y_seq = []
    
    # We iterate from lookback to len(data) - 1
    # because the delta at index i corresponds to change from i-1 to i.
    # To predict delta at i (using features up to i-1? or i?)
    # The original template used features at time t to predict delta at t.
    # So sequence ending at t-1 should predict delta at t? 
    # Or sequence ending at t (including current exog) predicts delta at t?
    # Original: row_exog = exog[t], target = delta[t].
    # This implies "Simultaneous" or "End-of-day" prediction where we know today's macro to predict today's close change.
    # Let's stick to that: Input window ending at index i (which has exog[i] and level[i]) predicts delta[i].
    # However, delta[i] needs level[i] and level[i-1]. Level[i] is in the input row.
    # This effectively means we are predicting the "change that happened to arrive at Level[i]"?
    # No, if we have Level[i] in the input, we can calculate the delta trivially.
    # 
    # Correction: 
    # We want to predict Level[t]. 
    # Input features available at t: Exog[t]. 
    # We do NOT have Level[t].
    # We DO have Level[t-1].
    # So Input Row t = [Level[t-1], Exog[t]].
    # Target = Level[t] (or Delta[t]).
    
    # Let's adjust data construction:
    # Shift Level back by 1 to use as feature.
    data_lagged_target = data.copy()
    data_lagged_target[f"{target_col}_lag1"] = data[target_col].shift(1)
    
    # Re-scale with the new column
    final_feature_cols = [c for c in feature_cols if c != target_col] + [f"{target_col}_lag1"]
    final_feature_cols = [c for c in data_lagged_target.columns if c in final_feature_cols]
    
    # Handle NaN from shift
    clean_data = data_lagged_target.dropna(subset=[f"{target_col}_lag1"] + final_feature_cols)
    
    # Refit scalers on clean data
    X_raw = clean_data[final_feature_cols].values
    y_raw_deltas = np.log(clean_data[target_col].values) - np.log(clean_data[f"{target_col}_lag1"].values)
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw_deltas.reshape(-1, 1)).flatten()
    
    # Build sequences
    # X_scaled shape: (N_samples, N_features)
    # We want (N_samples - lookback, lookback, N_features)
    # Input: t-lookback ... t-1. Target: t.
    
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i-lookback:i])
        y_seq.append(y_scaled[i])
        
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y, final_feature_cols


# -----------------------------
# PyTorch Model
# -----------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, 
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
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
        # x shape: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Use the output of the last time step
        out = out[:, -1, :]
        return self.fc(out)


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
    """Forecast test USDIDR recursively day-by-day using the GRU."""
    
    model.eval()
    preds = []
    
    # 1. Prepare initial history window
    # We need the last `lookback` rows from the training set
    # These rows must contain: Macro features + LAGGED USDIDR (which is the actual level at t-1)
    
    # Identify the target lag column name (it was engineered in prepare_sequences)
    # It should be f"{target_col}_lag1"
    lag_col = f"{target_col}_lag1"
    
    # Construct the full history of features (scaled)
    # We need to recreate the exact feature engineering steps on the combined set
    # to ensure consistency.
    
    # Start with engineering exogenous features for the full horizon
    full_exog = engineer_exogenous_features(combined_exog, infer_date_col(combined_exog))
    
    # Create the lagged target column
    # We start with train levels
    train_levels = pd.to_numeric(train_df[target_col], errors="coerce").astype(float).tolist()
    
    # We need a dataframe that aligns with full_exog rows to build the input tensor
    # Row i of input corresponds to time i.
    # Features at i = Exog[i] + Level[i-1].
    
    # Initialize the list of levels that we "know" or have predicted.
    # At index 0 of full_exog (which corresponds to train index 0), we don't have a previous level for input?
    # The training preparation dropped the first one.
    # Let's align indices.
    
    # full_exog length = len(train) + len(test)
    n_total = len(full_exog)
    
    # Array to store the 'known' levels for LAG feature construction
    # levels_known[i] is the level at time i.
    # Input features for time i use levels_known[i-1].
    levels_known = [np.nan] * (n_total + 1)
    for i in range(len(train_df)):
        levels_known[i] = train_levels[i]
        
    # We will fill levels_known for test indices as we predict.
    
    # Helper to get scaled input row for time t
    def get_scaled_row(t: int, current_levels: List[float]) -> np.ndarray:
        # t is index in full_exog
        # Input features: Exog[t] + Level[t-1]
        # Note: In prepare_sequences, we used Level[t] as target and Level[t-1] as feature.
        # So for predicting time t, we need:
        # 1. Exog at t (from full_exog)
        # 2. Level at t-1 (from current_levels)
        
        # Extract exog features (drop non-numeric or date if present)
        exog_row = full_exog.iloc[t]
        # We need to map feature_cols.
        # feature_cols contains names like "OIL", "GOLD", ..., f"{target_col}_lag1"
        
        row_dict = {}
        
        # Populate macro features
        # We must match the columns present in feature_cols
        for c in feature_cols:
            if c == lag_col:
                # This is the lagged target
                row_dict[c] = current_levels[t-1]
            else:
                # Exogenous feature
                if c in exog_row.index:
                    row_dict[c] = exog_row[c]
                else:
                    # Fallback for any mismatch
                    row_dict[c] = 0.0 
                    
        # Construct DataFrame in the correct column order
        row_df = pd.DataFrame([row_dict], columns=feature_cols)
        
        # Scale
        return scaler_X.transform(row_df.to_numpy())[0]

    # Build the initial window (the last `lookback` points of the training set)
    # The input for time `t` uses levels `t-1`.
    # The training set ends at index `len(train) - 1`.
    # We need to predict for test start index `len(train)`.
    # The window should end at `len(train) - 1` (input row for that time) or start earlier?
    # To predict test[0], we need a sequence of inputs ending at test[0]-1?
    # Original logic: Input row t -> Target t.
    # If we want to predict Test[0], we need Input Test[0].
    # Input Test[0] needs Level[Test[0]-1] (which is Train[-1]) and Exog[Test[0]].
    # So we construct the window of inputs for `Train[last_lookback]` ... `Test[0]`.
    # Wait, `Test[0]` is the first prediction step. We use `Test[0]` features to predict `Test[0]` target.
    # So the sequence is inputs for `t=Train_end-L` to `t=Test_start`.
    
    train_end_idx = len(train_df) - 1
    test_start_idx = len(train_df)
    
    # Create the initial sequence buffer
    # We need inputs from (test_start_idx - lookback) to (test_start_idx)
    # Actually, let's iterate and build a buffer of `lookback` inputs.
    
    # Let's pre-calculate inputs for the period we need.
    # We need the inputs for the last `lookback` steps of TRAIN to seed the GRU state,
    # plus the input for the first TEST step.
    # Sequence: Input(T-L), Input(T-L+1), ..., Input(T-1), Input(T). -> Pred(T).
    
    current_window = []
    
    # Build inputs for the tail of training set
    # We need inputs up to `train_end_idx`. 
    # Input `t` uses `levels[t-1]`.
    
    start_window_idx = train_end_idx - lookback + 1
    
    for i in range(start_window_idx, train_end_idx + 1):
        row_scaled = get_scaled_row(i, levels_known)
        current_window.append(row_scaled)
        
    # Now current_window contains inputs for indices [train_end-L+1, ..., train_end]
    # This corresponds to sequence inputs.
    # To predict `test_start_idx`, we append the input for `test_start_idx` to the sequence?
    # Yes. GRU processes sequence.
    # Seq: [Train_0, Train_1, ..., Train_Last, Test_0] -> Pred(Test_0).
    
    # Let's loop for predictions
    for i in range(len(test_df)):
        t = test_start_idx + i
        
        # 1. Get input features for time t
        # This requires Exog[t] (available) and Level[t-1] (known from previous step)
        input_row_t = get_scaled_row(t, levels_known)
        
        # 2. Append to current window
        current_window.append(input_row_t)
        if len(current_window) > lookback:
            current_window.pop(0)
            
        # 3. Predict
        X_tensor = torch.tensor([current_window], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_scaled_delta = model(X_tensor)
            pred_scaled_delta = pred_scaled_delta.cpu().numpy()[0, 0]
            
        # 4. Inverse transform
        pred_raw_delta = scaler_y.inverse_transform([[pred_scaled_delta]])[0, 0]
        
        # 5. Calculate Level
        # delta = log(Level[t]) - log(Level[t-1])
        # Level[t] = Level[t-1] * exp(delta)
        level_t = levels_known[t-1] * np.exp(pred_raw_delta)
        preds.append(level_t)
        
        # 6. Update history
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
    
    # Load data.
    train_raw = pd.read_csv(train_csv)
    test_raw = pd.read_csv(test_csv)
    actual_test_raw = pd.read_csv(actual_test_csv) if actual_test_csv is not None else None

    # Sort by date if possible.
    train_raw = sort_by_date_if_possible(train_raw)
    test_raw = sort_by_date_if_possible(test_raw)
    if actual_test_raw is not None:
        actual_test_raw = sort_by_date_if_possible(actual_test_raw)

    train_date_col = infer_date_col(train_raw)
    # test_date_col = infer_date_col(test_raw)

    # Align date columns / drop target from exogenous feature set.
    train_exog = train_raw.drop(columns=[TARGET_COL], errors="ignore")
    test_exog = test_raw.copy()

    # Combine train+test exogenous data so feature engineering is consistent
    combined_exog = pd.concat([train_exog, test_exog], ignore_index=True)
    effective_date_col = train_date_col
    combined_exog = engineer_exogenous_features(combined_exog, date_col=effective_date_col)

    # Build sequences for training
    # We only pass the training slice of combined_exog here
    X_train, y_train, scaler_X, scaler_y, feature_cols = prepare_sequences(
        train_df=train_raw,
        exog_df=combined_exog.iloc[: len(train_raw)].reset_index(drop=True),
        target_col=TARGET_COL,
        lookback=LOOKBACK,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Lookback window: {LOOKBACK}")
    print(f"Features per step: {len(feature_cols)}")

    # Convert to PyTorch Datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = GRUModel(input_size=len(feature_cols)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        
        # Simple early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Recursive forecasting for the test period.
    test_preds = forecast_test_period(
        model=model,
        train_df=train_raw,
        test_df=test_raw,
        combined_exog=combined_exog, # Pass full exog to access test features
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        lookback=LOOKBACK,
    )

    # Save submission.
    submission = pd.DataFrame({TARGET_COL: test_preds})
    
    # Try to recover date from test_raw
    test_date_col = infer_date_col(test_raw)
    if test_date_col and test_date_col in test_raw.columns:
        submission.insert(0, "Date", test_raw[test_date_col].values)
    
    submission_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_csv, index=False)
    print(f"Saved submission to: {submission_csv}")

    # Optional benchmark
    if actual_test_raw is not None:
        if TARGET_COL not in actual_test_raw.columns:
            raise ValueError(f"actual_test_csv must contain target column '{TARGET_COL}'.")
        actual = pd.to_numeric(actual_test_raw[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if len(actual) != len(test_preds):
            # Handle potential mismatch if filtering occurred
            min_len = min(len(actual), len(test_preds))
            actual = actual[:min_len]
            test_preds = test_preds[:min_len]
            print(f"Warning: Length mismatch, evaluating on first {min_len} samples.")

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