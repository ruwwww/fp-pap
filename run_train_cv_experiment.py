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
    
    # Standard ffill/bfill to remove NaNs in raw exogenous
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX", "BI_rate", "US_rate"]:
        if col in combined.columns:
            combined[col] = combined[col].ffill().bfill()
            
    # 1. Stationary Log-Returns
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # 2. Causal Lags including verified features
    combined["SP500_ret_lag1"] = combined["SP500_ret"].shift(1).fillna(0.0)
    combined["VIX_ret_lag1"] = combined["VIX_ret"].shift(1).fillna(0.0)
    combined["bi_rate_change_lag10"] = combined["bi_rate_change"].shift(10).fillna(0.0)
    combined["IHSG_ret_lag1"] = combined["IHSG_ret"].shift(1).fillna(0.0)
    combined["OIL_ret_lag1"] = combined["OIL_ret"].shift(1).fillna(0.0)
    combined["VIX_lag1"] = combined["VIX"].shift(1).fillna(15.0)
    
    return combined

def build_trend_table(train_df, combined, selected_lags, ade):
    levels = train_df["USDIDR"].astype(float).tolist()
    diffs = [levels[i] - levels[i - 1] for i in range(1, len(levels))]
    rows = []
    ys = []
    start = 252
    for t in range(start, len(train_df)):
        feats = ade.build_row_features(combined.iloc[t], levels[:t], diffs[: t - 1], selected_lags, [], "trend")
        rows.append(feats)
        ys.append(float(math.log(levels[t] / levels[t - 1])))
    X = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(ys, dtype=float)
    return X, y

def build_residual_table(train_df, combined, trend_model, trend_X, trend_y):
    trend_preds = trend_model.predict(trend_X)
    residuals = trend_y - trend_preds
    start = len(train_df) - len(trend_y)
    rows = []
    for t in range(start, len(train_df)):
        row_exog = combined.iloc[t]
        feats = {
            "SP500_ret_lag1": float(row_exog.get("SP500_ret_lag1", 0.0)),
            "VIX_ret_lag1": float(row_exog.get("VIX_ret_lag1", 0.0)),
            "bi_rate_change_lag10": float(row_exog.get("bi_rate_change_lag10", 0.0)),
            "IHSG_ret_lag1": float(row_exog.get("IHSG_ret_lag1", 0.0)),
            "OIL_ret_lag1": float(row_exog.get("OIL_ret_lag1", 0.0))
        }
        rows.append(feats)
    X = pd.DataFrame(rows)
    y = pd.Series(residuals)
    return X, y

def evaluate_on_split(train_df, val_df, combined, selected_lags, vix_fac, spread_fac, beta, ade, y_true_val=None):
    if y_true_val is not None:
        y_true = y_true_val
    else:
        y_true = val_df["USDIDR"].values
    
    # Train base trend model
    X_trend, y_trend = build_trend_table(train_df, combined, selected_lags, ade)
    trend_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]).fit(X_trend, y_trend)
    
    # Train Residual Shock model
    X_res, y_res = build_residual_table(train_df, combined, trend_model, X_trend, y_trend)
    res_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ]).fit(X_res, y_res)
    
    # Train Bias corrector on rolling segments of train_df
    levels = train_df["USDIDR"].astype(float).tolist()
    sim_len = len(val_df)
    
    # Use optimized segment size and step size for fast execution
    sim_len_segments = min(sim_len, 754)
    step_size = 252 # Increased from 63 to 252 (yields ~10 segments instead of 40)
    start_indices = list(range(252, len(train_df) - sim_len_segments, step_size))
    
    if len(start_indices) < 2:
        sim_len_segments = 252
        start_indices = list(range(252, len(train_df) - sim_len_segments, step_size))
        
    bias_samples = []
    
    # Extract scalers and models for fast predictions without pandas overhead
    scaler_trend = trend_model.named_steps["scaler"]
    ridge_trend = trend_model.named_steps["model"]
    trend_cols = list(trend_model.feature_names_in_)
    
    scaler_res = res_model.named_steps["scaler"]
    ridge_res = res_model.named_steps["model"]
    res_cols = list(res_model.feature_names_in_)
    
    for start_idx in start_indices:
        history_sim = list(levels[:start_idx])
        diffs_sim = [history_sim[j] - history_sim[j - 1] for j in range(1, len(history_sim))]
        preds_segment = []
        actuals_segment = levels[start_idx : start_idx + sim_len_segments]
        
        for k in range(sim_len_segments):
            t = start_idx + k
            row_exog = combined.iloc[t]
            
            # Trend prediction using fast list comprehension
            feats_trend = ade.build_row_features(row_exog, history_sim, diffs_sim, selected_lags, [], "trend")
            trend_vec = np.array([[feats_trend.get(col, 0.0) for col in trend_cols]])
            trend_scaled = scaler_trend.transform(trend_vec)
            ret_trend = float(ridge_trend.predict(trend_scaled)[0])
            
            # Shock prediction using fast list comprehension
            feats_res = {
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
                "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
                "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
            }
            res_vec = np.array([[feats_res.get(col, 0.0) for col in res_cols]])
            res_scaled = scaler_res.transform(res_vec)
            ret_shock = float(ridge_res.predict(res_scaled)[0])
            
            ret_total = ret_trend + ret_shock
            
            vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
            spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
            if ret_total > 0:
                if vix_lag1 > 14.0: ret_total *= vix_fac
                if spread < 0.8: ret_total *= spread_fac
                
            next_level = float(history_sim[-1] * math.exp(ret_total))
            preds_segment.append(next_level)
            history_sim.append(next_level)
            diffs_sim.append(next_level - history_sim[-2])
            
        errors_segment = np.array(actuals_segment) - np.array(preds_segment)
        for k in range(sim_len_segments - 4):
            t = start_idx + k
            row_exog = combined.iloc[t]
            target_bias = np.mean(errors_segment[k : k + 5])
            
            hist_window = preds_segment[max(0, k - 10) : k + 1]
            x_arr = np.arange(len(hist_window))
            current_level = preds_segment[k]
            slope = np.polyfit(x_arr, hist_window, 1)[0] / current_level if len(hist_window) >= 3 else 0.0
            curvature = np.polyfit(x_arr, hist_window, 2)[0] / current_level if len(hist_window) >= 3 else 0.0
            recent_direction = (preds_segment[k] - preds_segment[max(0, k-5)]) / preds_segment[max(0, k-5)]
            
            feats_bias = {
                "forecast_age": float(k),
                "trend_slope": slope,
                "trend_curvature": curvature,
                "VIX": float(row_exog.get("VIX", 18.0)),
                "recent_forecast_direction": recent_direction,
                "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
                "SP500_ret": float(row_exog.get("SP500_ret", 0.0)),
                "target_bias": target_bias
            }
            bias_samples.append(feats_bias)
            
    df_bias = pd.DataFrame(bias_samples)
    stable_features = ["forecast_age", "trend_slope", "trend_curvature", "VIX", "recent_forecast_direction", "GOLD_ret", "SP500_ret"]
    
    bias_models = {}
    regime_conditions = {
        "VIX_Low": df_bias["VIX"] < 14.0,
        "VIX_Med": (df_bias["VIX"] >= 14.0) & (df_bias["VIX"] <= 20.0),
        "VIX_High": df_bias["VIX"] > 20.0
    }
    
    for rname, condition in regime_conditions.items():
        sub = df_bias[condition]
        model_b = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=100.0))
        ]).fit(sub[stable_features], sub["target_bias"])
        bias_models[rname] = model_b
        
    # Out-of-Sample validation simulation
    history = train_df["USDIDR"].astype(float).tolist()
    history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    
    preds_base_list = []
    preds_final = []
    split_idx = len(train_df)
    
    # Extract model pipelines for fast validation run
    scaler_bias_low = bias_models["VIX_Low"].named_steps["scaler"]
    ridge_bias_low = bias_models["VIX_Low"].named_steps["model"]
    
    scaler_bias_med = bias_models["VIX_Med"].named_steps["scaler"]
    ridge_bias_med = bias_models["VIX_Med"].named_steps["model"]
    
    scaler_bias_high = bias_models["VIX_High"].named_steps["scaler"]
    ridge_bias_high = bias_models["VIX_High"].named_steps["model"]
    
    for i in range(len(val_df)):
        idx = split_idx + i
        row_exog = combined.iloc[idx]
        
        feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
        trend_vec = np.array([[feats_trend.get(col, 0.0) for col in trend_cols]])
        trend_scaled = scaler_trend.transform(trend_vec)
        ret_trend = float(ridge_trend.predict(trend_scaled)[0])
        
        feats_res = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
            "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
            "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
        }
        res_vec = np.array([[feats_res.get(col, 0.0) for col in res_cols]])
        res_scaled = scaler_res.transform(res_vec)
        ret_shock = float(ridge_res.predict(res_scaled)[0])
        
        ret_total = ret_trend + ret_shock
        
        vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
        spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
        if ret_total > 0:
            if vix_lag1 > 14.0: ret_total *= vix_fac
            if spread < 0.8: ret_total *= spread_fac
            
        pred_base = float(history[-1] * math.exp(ret_total))
        preds_base_list.append(pred_base)
        history.append(pred_base)
        history_diffs.append(pred_base - history[-2])
        
        hist_window = preds_base_list[max(0, i - 10) : i + 1]
        x_arr = np.arange(len(hist_window))
        slope = np.polyfit(x_arr, hist_window, 1)[0] / pred_base if len(hist_window) >= 3 else 0.0
        curvature = np.polyfit(x_arr, hist_window, 2)[0] / pred_base if len(hist_window) >= 3 else 0.0
        recent_direction = (preds_base_list[i] - preds_base_list[max(0, i-5)]) / preds_base_list[max(0, i-5)]
        
        bias_feats = {
            "forecast_age": float(i),
            "trend_slope": slope,
            "trend_curvature": curvature,
            "VIX": float(row_exog.get("VIX", 18.0)),
            "recent_forecast_direction": recent_direction,
            "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
            "SP500_ret": float(row_exog.get("SP500_ret", 0.0))
        }
        bias_vec = np.array([[bias_feats.get(col, 0.0) for col in stable_features]])
        
        if vix_lag1 < 14.0:
            bias_scaled = scaler_bias_low.transform(bias_vec)
            bias_correction = float(ridge_bias_low.predict(bias_scaled)[0])
        elif 14.0 <= vix_lag1 <= 20.0:
            bias_scaled = scaler_bias_med.transform(bias_vec)
            bias_correction = float(ridge_bias_med.predict(bias_scaled)[0])
        else:
            bias_scaled = scaler_bias_high.transform(bias_vec)
            bias_correction = float(ridge_bias_high.predict(bias_scaled)[0])
            
        preds_final.append(pred_base + beta * bias_correction)
        
    return rmse(y_true, preds_final)

def main():
    train_raw, test_exog, test_actual = load_data()
    y_true = test_actual["USDIDR"].astype(float).to_numpy()
    
    combined = prepare_combined_data(train_raw, test_exog)
    
    import assumption_driven_experiment as ade
    combined = ade.make_causal_exog(combined)
    combined = combined.fillna(0.0)
    
    # 3 Folds of Walk-Forward validation on Train Set
    folds = [
        (1221, 754),
        (1975, 754),
        (2729, 752)
    ]
    
    # Fast grid search space (12 combinations)
    betas = [0.0, 0.1, 0.2, 0.25]
    vix_factors = [1.05, 1.10]
    spread_factors = [1.02, 1.06]
    
    selected_lags = [1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]
    
    print("--- Running FAST Time-Series CV on Train Set (Target: OOS RMSE < 290) ---")
    best_avg_rmse = 99999.0
    best_params = None
    
    for vix_fac in vix_factors:
        for spread_fac in spread_factors:
            for beta in betas:
                fold_rmses = []
                for train_len, val_len in folds:
                    train_fold = combined.iloc[:train_len].reset_index(drop=True)
                    val_fold = combined.iloc[train_len : train_len + val_len].reset_index(drop=True)
                    
                    try:
                        score = evaluate_on_split(train_fold, val_fold, combined, selected_lags, vix_fac, spread_fac, beta, ade)
                        fold_rmses.append(score)
                    except Exception as e:
                        fold_rmses.append(99999.0)
                        
                avg_rmse = np.mean(fold_rmses)
                print(f"vix={vix_fac}, spread={spread_fac}, beta={beta} -> Avg Train CV RMSE: {avg_rmse:.2f}")
                
                if avg_rmse < best_avg_rmse:
                    best_avg_rmse = avg_rmse
                    best_params = (vix_fac, spread_fac, beta)
                    
    best_vix, best_spread, best_beta = best_params
    print("\n=== OPTIMAL PARAMETERS CHOSEN STRICTLY FROM TRAIN CV ===")
    print(f"  VIX Gate Factor      : {best_vix}")
    print(f"  Spread Gate Factor   : {best_spread}")
    print(f"  Bias Shrinkage (beta): {best_beta}")
    print(f"  Average Train CV RMSE: {best_avg_rmse:.2f}")
    
    # Train final model on FULL data_train using CV parameters
    print("\nFitting final model on full training set using optimal CV hyperparameters...")
    train_df = combined.iloc[:len(train_raw)].reset_index(drop=True)
    test_df = combined.iloc[len(train_raw):].reset_index(drop=True)
    
    final_oos_rmse = evaluate_on_split(train_df, test_df, combined, selected_lags, best_vix, best_spread, best_beta, ade, y_true_val=y_true)
    print(f"\n==========================================")
    print(f"  FINAL HOLDOUT TEST OOS RMSE: {final_oos_rmse:.4f}")
    print(f"==========================================")
    
    # Save final predictions to submission.csv
    # Re-run simulation to get prediction list
    X_trend, y_trend = build_trend_table(train_df, combined, selected_lags, ade)
    trend_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]).fit(X_trend, y_trend)
    X_res, y_res = build_residual_table(train_df, combined, trend_model, X_trend, y_trend)
    res_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]).fit(X_res, y_res)
    
    # Train Bias models on full training data
    levels = train_df["USDIDR"].astype(float).tolist()
    sim_len_segments = 754
    step_size = 126
    start_indices = list(range(252, len(train_df) - sim_len_segments, step_size))
    bias_samples = []
    
    trend_cols = trend_model.feature_names_in_
    res_cols = res_model.feature_names_in_
    
    scaler_trend = trend_model.named_steps["scaler"]
    ridge_trend = trend_model.named_steps["model"]
    trend_cols = list(trend_model.feature_names_in_)
    
    scaler_res = res_model.named_steps["scaler"]
    ridge_res = res_model.named_steps["model"]
    res_cols = list(res_model.feature_names_in_)
    
    for start_idx in start_indices:
        history_sim = list(levels[:start_idx])
        diffs_sim = [history_sim[j] - history_sim[j - 1] for j in range(1, len(history_sim))]
        preds_segment = []
        actuals_segment = levels[start_idx : start_idx + sim_len_segments]
        
        for k in range(sim_len_segments):
            t = start_idx + k
            row_exog = combined.iloc[t]
            feats_trend = ade.build_row_features(row_exog, history_sim, diffs_sim, selected_lags, [], "trend")
            trend_vec = np.array([[feats_trend.get(col, 0.0) for col in trend_cols]])
            trend_scaled = scaler_trend.transform(trend_vec)
            ret_trend = float(ridge_trend.predict(trend_scaled)[0])
            
            feats_res = {
                "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
                "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
                "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
                "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
                "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
            }
            res_vec = np.array([[feats_res.get(col, 0.0) for col in res_cols]])
            res_scaled = scaler_res.transform(res_vec)
            ret_shock = float(ridge_res.predict(res_scaled)[0])
            
            ret_total = ret_trend + ret_shock
            vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
            spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
            if ret_total > 0:
                if vix_lag1 > 14.0: ret_total *= best_vix
                if spread < 0.8: ret_total *= best_spread
            next_level = float(history_sim[-1] * math.exp(ret_total))
            preds_segment.append(next_level)
            history_sim.append(next_level)
            diffs_sim.append(next_level - history_sim[-2])
            
        errors_segment = np.array(actuals_segment) - np.array(preds_segment)
        for k in range(sim_len_segments - 4):
            t = start_idx + k
            row_exog = combined.iloc[t]
            target_bias = np.mean(errors_segment[k : k + 5])
            hist_window = preds_segment[max(0, k - 10) : k + 1]
            x_arr = np.arange(len(hist_window))
            current_level = preds_segment[k]
            slope = np.polyfit(x_arr, hist_window, 1)[0] / current_level if len(hist_window) >= 3 else 0.0
            curvature = np.polyfit(x_arr, hist_window, 2)[0] / current_level if len(hist_window) >= 3 else 0.0
            recent_direction = (preds_segment[k] - preds_segment[max(0, k-5)]) / preds_segment[max(0, k-5)]
            
            feats_bias = {
                "forecast_age": float(k),
                "trend_slope": slope,
                "trend_curvature": curvature,
                "VIX": float(row_exog.get("VIX", 18.0)),
                "recent_forecast_direction": recent_direction,
                "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
                "SP500_ret": float(row_exog.get("SP500_ret", 0.0)),
                "target_bias": target_bias
            }
            bias_samples.append(feats_bias)
            
    df_bias = pd.DataFrame(bias_samples)
    stable_features = ["forecast_age", "trend_slope", "trend_curvature", "VIX", "recent_forecast_direction", "GOLD_ret", "SP500_ret"]
    bias_models = {}
    regime_conditions = {
        "VIX_Low": df_bias["VIX"] < 14.0,
        "VIX_Med": (df_bias["VIX"] >= 14.0) & (df_bias["VIX"] <= 20.0),
        "VIX_High": df_bias["VIX"] > 20.0
    }
    for rname, condition in regime_conditions.items():
        sub = df_bias[condition]
        model_b = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=100.0))]).fit(sub[stable_features], sub["target_bias"])
        bias_models[rname] = model_b
        
    # Generate final test predictions
    history = train_df["USDIDR"].astype(float).tolist()
    history_diffs = [history[i] - history[i - 1] for i in range(1, len(history))]
    preds_base_list = []
    preds_final = []
    
    scaler_bias_low = bias_models["VIX_Low"].named_steps["scaler"]
    ridge_bias_low = bias_models["VIX_Low"].named_steps["model"]
    
    scaler_bias_med = bias_models["VIX_Med"].named_steps["scaler"]
    ridge_bias_med = bias_models["VIX_Med"].named_steps["model"]
    
    scaler_bias_high = bias_models["VIX_High"].named_steps["scaler"]
    ridge_bias_high = bias_models["VIX_High"].named_steps["model"]
    
    for i in range(len(test_df)):
        idx = len(train_raw) + i
        row_exog = combined.iloc[idx]
        feats_trend = ade.build_row_features(row_exog, history, history_diffs, selected_lags, [], "trend")
        trend_vec = np.array([[feats_trend.get(col, 0.0) for col in trend_cols]])
        trend_scaled = scaler_trend.transform(trend_vec)
        ret_trend = float(ridge_trend.predict(trend_scaled)[0])
        
        feats_res = {
            "SP500_ret_lag1": row_exog["SP500_ret_lag1"],
            "VIX_ret_lag1": row_exog["VIX_ret_lag1"],
            "bi_rate_change_lag10": row_exog["bi_rate_change_lag10"],
            "IHSG_ret_lag1": row_exog["IHSG_ret_lag1"],
            "OIL_ret_lag1": row_exog["OIL_ret_lag1"]
        }
        res_vec = np.array([[feats_res.get(col, 0.0) for col in res_cols]])
        res_scaled = scaler_res.transform(res_vec)
        ret_shock = float(ridge_res.predict(res_scaled)[0])
        
        ret_total = ret_trend + ret_shock
        vix_lag1 = float(row_exog.get("VIX_lag1", 15.0))
        spread = float(row_exog.get("BI_rate", 5.75)) - float(row_exog.get("US_rate", 5.08))
        if ret_total > 0:
            if vix_lag1 > 14.0: ret_total *= best_vix
            if spread < 0.8: ret_total *= best_spread
        pred_base = float(history[-1] * math.exp(ret_total))
        preds_base_list.append(pred_base)
        history.append(pred_base)
        history_diffs.append(pred_base - history[-2])
        
        hist_window = preds_base_list[max(0, i - 10) : i + 1]
        x_arr = np.arange(len(hist_window))
        slope = np.polyfit(x_arr, hist_window, 1)[0] / pred_base if len(hist_window) >= 3 else 0.0
        curvature = np.polyfit(x_arr, hist_window, 2)[0] / pred_base if len(hist_window) >= 3 else 0.0
        recent_direction = (preds_base_list[i] - preds_base_list[max(0, i-5)]) / preds_base_list[max(0, i-5)]
        
        bias_feats = {
            "forecast_age": float(i),
            "trend_slope": slope,
            "trend_curvature": curvature,
            "VIX": float(row_exog.get("VIX", 18.0)),
            "recent_forecast_direction": recent_direction,
            "GOLD_ret": float(row_exog.get("GOLD_ret", 0.0)),
            "SP500_ret": float(row_exog.get("SP500_ret", 0.0))
        }
        bias_vec = np.array([[bias_feats.get(col, 0.0) for col in stable_features]])
        
        if vix_lag1 < 14.0:
            bias_scaled = scaler_bias_low.transform(bias_vec)
            bias_correction = float(ridge_bias_low.predict(bias_scaled)[0])
        elif 14.0 <= vix_lag1 <= 20.0:
            bias_scaled = scaler_bias_med.transform(bias_vec)
            bias_correction = float(ridge_bias_med.predict(bias_scaled)[0])
        else:
            bias_scaled = scaler_bias_high.transform(bias_vec)
            bias_correction = float(ridge_bias_high.predict(bias_scaled)[0])
            
        preds_final.append(pred_base + best_beta * bias_correction)
        
    preds_final = np.array(preds_final)
    
    # Save best predicted to submission.csv
    pd.DataFrame({
        "Date": test_actual["Date"],
        "USDIDR": preds_final
    }).to_csv("submission.csv", index=False)
    print("Predictions saved to 'submission.csv'!")
    
    # Plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 6))
    plt.plot(test_actual["Date"], y_true, color="black", label="Actual")
    plt.plot(test_actual["Date"], preds_final, color="red", label=f"3-Layer Gated CV (RMSE={final_oos_rmse:.2f})")
    plt.title("USDIDR Out-Of-Sample Forecasting: 3-Layer Gated CV System")
    plt.legend()
    plt.savefig("final_fluctuating_macro_plot.png", dpi=150)
    plt.close()
    print("Updated 'final_fluctuating_macro_plot.png' successfully!")

if __name__ == "__main__":
    main()
