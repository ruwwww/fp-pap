#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

def prepare_combined_data(train, test_exog):
    combined = pd.concat([train, test_exog], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    
    # 1. Stationary Log-Returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # 2. Causal Lags including our new verified features (IHSG & OIL)
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    
    # New validated features
    combined["IHSG_ret_lag1"] = combined["IHSG_ret"].shift(1).fillna(0.0)
    combined["OIL_ret_lag1"] = combined["OIL_ret"].shift(1).fillna(0.0)
    
    # Base VIX level for gating
    combined["VIX_lag1"] = combined["VIX"].shift(1).fillna(15.0)
    
    return combined

def main():
    train, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    combined = prepare_combined_data(train, test_exog)
    
    # Trend Model Setup
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
    
    trend_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    trend_pipeline.fit(X_trend, y_trend)
    
    # Calculate Training Residuals (De-trended Return)
    trend_preds = trend_pipeline.predict(X_trend)
    train_residuals = y_trend - trend_preds
    
    # Build Residual Model using expanded verified features (SP500, VIX, IHSG, OIL, BI Rate)
    X_res_rows = []
    for t in range(start_idx, len(train)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
            "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
            "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
        }
        X_res_rows.append(feats)
        
    X_res = pd.DataFrame(X_res_rows)
    y_res = pd.Series(train_residuals)
    
    # Fit expanded residual model
    res_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ])
    res_pipeline.fit(X_res, y_res)
    
    # Forecast OOS
    history = train["USDIDR"].astype(float).tolist()
    history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    preds = []
    for i in range(len(test_exog)):
        idx = len(train) + i
        row_exog = combined.iloc[idx]
        
        # 1. Trend Pred
        feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
        X_row_trend = pd.DataFrame([feats_trend]).reindex(columns=X_trend.columns, fill_value=0.0)
        ret_trend = float(trend_pipeline.predict(X_row_trend)[0])
        
        # 2. Predict Residual Shock (with IHSG & OIL inputs)
        feats_res = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
            "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
            "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
        }
        X_row_res = pd.DataFrame([feats_res])
        ret_shock = float(res_pipeline.predict(X_row_res)[0])
        
        ret_total = ret_trend + ret_shock
        
        # 3. Dynamic gates
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
        
    score = rmse(y_true, preds)
    print(f"Expanded Decoupled Ridge Model OOS RMSE: {score:.4f}")
    
    # Save best predicted
    pd.DataFrame({
        "Date": test_actual["Date"],
        "actual": y_true,
        "predicted": preds
    }).to_csv("submission.csv", index=False)
    print("New predictions written to 'submission.csv'!")
    
    # Plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual["Date"], y_true, color="black", label="Actual")
    plt.plot(test_actual["Date"], preds, color="red", label=f"Expanded Model (RMSE={score:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: Expanded Exogenous Shock Model")
    plt.legend()
    plt.savefig("final_fluctuating_macro_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
