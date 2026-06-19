#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def main():
    train = pd.read_csv("data_train.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    
    # We must handle business days missing value gap for seasonal decomposition
    # Set date as index and resample to business day frequency
    train.set_index("Date", inplace=True)
    ts = train["USDIDR"].resample("B").ffill()
    
    # 1. Classical Seasonal Decomposition (Weekly Cycle = 5 business days)
    print("=== SEASONAL & CYCLIC DECOMPOSITION ANALYSIS (Weekly Cycle = 5) ===")
    decomp_weekly = seasonal_decompose(ts, model="additive", period=5)
    
    weekly_seasonal = decomp_weekly.seasonal
    weekly_trend = decomp_weekly.trend
    weekly_resid = decomp_weekly.resid.dropna()
    
    # Calculate variance contribution of each component
    var_total = np.var(ts)
    var_trend = np.var(weekly_trend.dropna())
    var_season = np.var(weekly_seasonal)
    var_resid = np.var(weekly_resid)
    
    print(f"Total USDIDR Level Variance: {var_total:.2f}")
    print(f"Trend Component Variance:     {var_trend:.2f} ({var_trend/var_total*100:.2f}%)")
    print(f"Weekly Seasonal Variance:     {var_season:.2f} ({var_season/var_total*100:.2f}%)")
    print(f"Residual Noise Variance:      {var_resid:.2f} ({var_resid/var_total*100:.2f}%)")
    
    # 2. Monthly Cycle Decomposition (period = 21 business days)
    print("\n=== MONTHLY CYCLE DECOMPOSITION ANALYSIS (Period = 21) ===")
    decomp_monthly = seasonal_decompose(ts, model="additive", period=21)
    
    monthly_seasonal = decomp_monthly.seasonal
    monthly_trend = decomp_monthly.trend
    monthly_resid = decomp_monthly.resid.dropna()
    
    var_m_trend = np.var(monthly_trend.dropna())
    var_m_season = np.var(monthly_seasonal)
    var_m_resid = np.var(monthly_resid)
    
    print(f"Monthly Seasonal Variance:    {var_m_season:.2f} ({var_m_season/var_total*100:.2f}%)")
    print(f"Residual Noise Variance:      {var_m_resid:.2f} ({var_m_resid/var_total*100:.2f}%)")
    
    # Test stationarity of the residuals after removing seasonal + trend
    adf_res = adfuller(weekly_resid)
    print(f"\nADF on Weekly De-seasoned Residuals: stat={adf_res[0]:.4f}, p-value={adf_res[1]:.4e}")
    
    # Save a diagnostic plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(ts, label="Original USDIDR", color="black")
    axes[0].legend(loc="upper left")
    axes[0].set_title("USDIDR Time Series Decomposition")
    
    axes[1].plot(weekly_trend, label="Trend Component", color="blue")
    axes[1].legend(loc="upper left")
    
    axes[2].plot(weekly_seasonal.head(100), label="Weekly Seasonal Pattern (First 100 days shown)", color="orange")
    axes[2].legend(loc="upper left")
    
    axes[3].plot(weekly_resid, label="De-seasoned Residuals", color="red", alpha=0.6)
    axes[3].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("seasonal_decomposition_plot.png", dpi=150)
    plt.close()
    print("\nDecomposition plot saved to 'seasonal_decomposition_plot.png'")

if __name__ == "__main__":
    main()
