import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

def run_analysis():
    # Load data
    train = pd.read_csv("data_train.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    
    # Calculate log returns of USDIDR
    train["log_ret"] = np.log(train["USDIDR"]).diff().fillna(0.0)
    
    # Simple linear detrending or MA detrending on target level
    # 1. Linear Detrending
    x = np.arange(len(train))
    slope, intercept = np.polyfit(x, train["USDIDR"], 1)
    train["linear_trend"] = slope * x + intercept
    train["detrended_level"] = train["USDIDR"] - train["linear_trend"]
    
    # 2. Log-Return residuals (deviation from historical mean return)
    mean_ret = train["log_ret"].mean()
    train["detrended_ret"] = train["log_ret"] - mean_ret
    
    print("--- Test Stationarity on Detrended Series ---")
    adf_level = adfuller(train["detrended_level"])
    print(f"ADF Detrended Level: stat={adf_level[0]:.4f}, p-value={adf_level[1]:.4e}")
    
    adf_ret = adfuller(train["detrended_ret"])
    print(f"ADF Detrended Log-Return: stat={adf_ret[0]:.4f}, p-value={adf_ret[1]:.4e}")
    
    # Ljung-Box Test on detrended log-return to see if it's pure noise or has structure
    print("\n--- Ljung-Box autocorrelation test on Detrended Log-Return ---")
    lb_res = acorr_ljungbox(train["detrended_ret"], lags=[1, 2, 5, 10, 20], return_df=True)
    print(lb_res)
    
    # Check correlation of detrended log-return with lag exogenous variables
    train["SP500_ret"] = np.log(train["SP500"]).diff().fillna(0.0)
    train["VIX_ret"] = np.log(train["VIX"]).diff().fillna(0.0)
    train["BI_rate_diff"] = train["BI_rate"].diff().fillna(0.0)
    
    print("\n--- Correlation of Detrended Log-Return with Exogenous Shocks ---")
    for col in ["SP500_ret", "VIX_ret", "BI_rate_diff"]:
        corr = train["detrended_ret"].corr(train[col].shift(1))
        print(f"Corr with {col} (lag 1): {corr:.4f}")

if __name__ == "__main__":
    run_analysis()
