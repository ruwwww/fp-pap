#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import math
import warnings
warnings.filterwarnings("ignore")

# Define RMSE
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))

def load_data():
    train = pd.read_csv("data_train.csv")
    test_exog = pd.read_csv("data_test.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    return train, test_exog, test_actual

def main():
    train, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    # 1. Prepare features
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # Stationary Exogenous returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # Lagged features matching step 2 diagnostics
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    # Selected Trend Lags from PACF
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    levels = train["USDIDR"].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    
    import assumption_driven_experiment as ade
    X_trend_rows = []
    y_trend = []
    start_idx = max(max(selected_lags), 252)
    
    for t in range(start_idx, len(train)):
        row_exog = combined.iloc[t]
        feats = ade.build_row_features(row_exog, levels[:t], diffs[:t-1], selected_lags, [], "trend")
        X_trend_rows.append(feats)
        y_trend.append(float(math.log(levels[t] / levels[t - 1])))
        
    X_trend = pd.DataFrame(X_trend_rows).fillna(0.0)
    y_trend = pd.Series(y_trend)
    
    # Fit Trend Model
    from sklearn.linear_model import Ridge
    trend_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_pipeline.fit(X_trend, y_trend)
    
    # Calculate Training Residuals (De-trended log-returns)
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
    # 2. Build Residual Feature Matrix (X_res) for LightGBM
    X_res_rows = []
    for t in range(start_idx, len(train)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
        }
        X_res_rows.append(feats)
    X_res = pd.DataFrame(X_res_rows)
    y_res = pd.Series(train_residuals)
    
    # Fit LightGBM on decoupled residuals using Huber objective (to handle fat-tails Kurtosis 6.02)
    # We restrict n_estimators and depth to strictly prevent OOS recursive feedback explosion
    lgb_res = lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9, # Huber threshold
        n_estimators=30,
        max_depth=3,
        learning_rate=0.03,
        random_state=42,
        verbose=-1
    )
    lgb_res.fit(X_res, y_res)
    
    # 3. Forecast OOS recursively
    history = train["USDIDR"].astype(float).tolist()
    history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    preds = []
    for i in range(len(test_exog)):
        idx = len(train) + i
        row_exog = combined.iloc[idx]
        
        # Trend Prediction
        feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
        ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
        
        # LightGBM Residual Prediction (decoupled)
        feats_res = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"]
        }
        X_row_res = pd.DataFrame([feats_res])
        ret_shock = float(lgb_res.predict(X_row_res)[0])
        
        ret_total = ret_trend + ret_shock
        
        # Dynamic Risk-gates
        vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
        bi_rate = float(row_exog.get("BI_rate", 5.75))
        us_rate = float(row_exog.get("US_rate", 5.08))
        spread = bi_rate - us_rate
        
        if ret_total > 0:
            if vix_lag1 > 14.0:
                ret_total *= 1.10
            if spread < 0.8:
                ret_total *= 1.06
                
        next_level = float(history[-1] * math.exp(ret_total))
        preds.append(next_level)
        history.append(next_level)
        history_diffs.append(next_level - history[-2])
        
    lgb_rmse = rmse(y_true, preds)
    print(f"Decoupled LightGBM (Huber Loss) OOS RMSE: {lgb_rmse:.4f}")
    
    # Save predictions
    pd.DataFrame({
        "Date": test_actual["Date"],
        "actual": y_true,
        "predicted": preds
    }).to_csv("decoupled_lgbm_predictions.csv", index=False)
    
    # Plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual["Date"], y_true, color="black", label="Actual")
    plt.plot(test_actual["Date"], preds, color="red", label=f"Decoupled LightGBM (RMSE={lgb_rmse:.2f})")
    plt.title("USDIDR OOS Forecast: Two-Stage Decoupled LightGBM with Huber Loss")
    plt.legend()
    plt.savefig("decoupled_lgbm_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
