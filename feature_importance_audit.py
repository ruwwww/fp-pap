#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def main():
    train = pd.read_csv("data_train.csv")
    
    # Calculate target log returns
    train["log_ret"] = np.log(train["USDIDR"]).diff().fillna(0.0)
    
    # 1. Feature Engineering: Create all potential candidate features
    # A. Exogenous levels and returns
    exogs = ["OIL", "GOLD", "SP500", "IHSG", "VIX", "CPI", "BI_rate", "US_rate"]
    for col in exogs:
        train[f"{col}_ret"] = np.log(train[col]).diff().fillna(0.0) if col not in ["BI_rate", "US_rate", "CPI"] else train[col].diff().fillna(0.0)
        train[f"{col}_lag1"] = train[f"{col}_ret"].shift(1)
        train[f"{col}_lag2"] = train[f"{col}_ret"].shift(2)
        
    # B. Yield Spread & changes
    train["spread"] = train["BI_rate"] - train["US_rate"]
    train["spread_change"] = train["spread"].diff().fillna(0.0)
    train["spread_lag1"] = train["spread_change"].shift(1)
    
    # C. Calendar Features (Just to double check)
    train["Date"] = pd.to_datetime(train["Date"])
    train["day_of_week"] = train["Date"].dt.dayofweek
    train["month"] = train["Date"].dt.month
    
    # Drop NaNs created by lagging
    train.dropna(inplace=True)
    
    # Target and predictor setup
    y = train["log_ret"]
    feature_cols = [c for c in train.columns if "lag" in c or "change" in c or "spread" in c or c in ["day_of_week", "month"]]
    X = train[feature_cols]
    
    # 2. Fit LightGBM to identify feature importance
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    print("=== RANKING IMPORTANCE FITUR KANDIDAT (LIGHTGBM) ===")
    print(importance.head(20))

if __name__ == "__main__":
    main()
