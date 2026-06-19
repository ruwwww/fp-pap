#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import jarque_bera, skew, kurtosis
import os

def main():
    # Load data
    train = pd.read_csv("data_train.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    
    # 1. Separate Smooth Linear Trend
    # Fit simple linear trend on Train USDIDR
    x_train = np.arange(len(train))
    slope, intercept = np.polyfit(x_train, train["USDIDR"], 1)
    
    train["smooth_trend"] = slope * x_train + intercept
    train["de_trended"] = train["USDIDR"] - train["smooth_trend"]
    
    print("=== 1. UJI STASIONERITAS (ADF TEST) ===")
    adf_res = adfuller(train["de_trended"])
    print(f"ADF Statistic: {adf_res[0]:.4f}")
    print(f"p-value: {adf_res[1]:.4e}")
    print("Interpretasi: Data de-trended level (USDIDR - Linear Trend) " + 
          ("STASIONER" if adf_res[1] < 0.05 else "TIDAK STASIONER") + 
          " pada level 5% significance.")
    
    # Let's also check log-return de-trend (which we use in our quantitative modeling)
    train["log_ret"] = np.log(train["USDIDR"]).diff().fillna(0.0)
    train["de_trended_ret"] = train["log_ret"] - train["log_ret"].mean()
    adf_ret = adfuller(train["de_trended_ret"])
    print(f"\nADF on De-trended Log-Return: stat={adf_ret[0]:.4f}, p-value={adf_ret[1]:.4e}")
    
    print("\n=== 2. ANALISIS AUTOKORELASI (ACF & PACF) ===")
    lag_acf = acf(train["de_trended"], nlags=40)
    lag_pacf = pacf(train["de_trended"], nlags=40, method="ywm")
    
    # Identify significant lags (outside confidence interval)
    conf_interval = 1.96 / np.sqrt(len(train))
    sig_acf_lags = [i for i, val in enumerate(lag_acf) if abs(val) > conf_interval and i > 0]
    sig_pacf_lags = [i for i, val in enumerate(lag_pacf) if abs(val) > conf_interval and i > 0]
    
    print(f"Batas Signifikansi (95% CI): +/- {conf_interval:.4f}")
    print(f"Significant Lags in ACF (first 10): {sig_acf_lags[:10]}")
    print(f"Significant Lags in PACF (first 10): {sig_pacf_lags[:10]}")
    
    print("\n=== 3. ANALISIS KORELASI DENGAN VARIABEL MAKRO (CROSS-CORRELATION) ===")
    # Calculate returns of exogenous
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        train[f"{col}_ret"] = np.log(train[col]).diff().fillna(0.0)
    train["BI_rate_diff"] = train["BI_rate"].diff().fillna(0.0)
    
    for exog in ["SP500_ret", "VIX_ret", "BI_rate_diff"]:
        corrs = []
        for lag in range(0, 11):
            corr = train["de_trended_ret"].corr(train[exog].shift(lag))
            corrs.append((lag, corr))
        # Find max absolute correlation
        best_lag, best_corr = max(corrs, key=lambda x: abs(x[1]))
        print(f"CCF {exog} -> De-trended Return: Best Lag={best_lag} dengan Korelasi={best_corr:.4f}")
        
    print("\n=== 4. TRANSFORMASI FITUR UNTUK MODEL NON-LINEAR ===")
    # Create rolling volatility and rolling mean for the de-trended return series
    train["rolling_vol_7d"] = train["de_trended_ret"].rolling(window=7).std().fillna(0.0)
    train["rolling_vol_30d"] = train["de_trended_ret"].rolling(window=30).std().fillna(0.0)
    train["rolling_mean_7d"] = train["de_trended_ret"].rolling(window=7).mean().fillna(0.0)
    
    print(f"Rata-rata Volatilitas 7-hari: {train['rolling_vol_7d'].mean():.6f}")
    print(f"Rata-rata Volatilitas 30-hari: {train['rolling_vol_30d'].mean():.6f}")
    
    print("\n=== 5. EVALUASI DISTRIBUSI DE-TRENDED ===")
    s = skew(train["de_trended_ret"])
    k = kurtosis(train["de_trended_ret"])
    jb_stat, jb_p = jarque_bera(train["de_trended_ret"])
    
    print(f"Skewness: {s:.4f} (Positif/Negatif Tail)")
    print(f"Kurtosis: {k:.4f} (Fat Tails check, normal=0 dalam scipy excess kurtosis)")
    print(f"Jarque-Bera Test: statistic={jb_stat:.4f}, p-value={jb_p:.4e}")
    print("Interpretasi Distribusi: " + 
          ("Memiliki ekor tebal / Leptokurtik (Fat-tailed) & Non-Normal." if k > 0 and jb_p < 0.05 else "Mendekati Normal."))

    # Generate ACF/PACF and Distribution Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # De-trended level chart
    axes[0, 0].plot(train["de_trended"], color="purple", alpha=0.8)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title("1. De-trended Level USDIDR (Harga - Tren Linear)")
    
    # ACF / PACF Plot
    axes[0, 1].bar(range(1, 41), lag_pacf[1:41], color="blue", alpha=0.7)
    axes[0, 1].axhline(conf_interval, color="red", linestyle="--")
    axes[0, 1].axhline(-conf_interval, color="red", linestyle="--")
    axes[0, 1].set_title("2. Partial Autocorrelation Function (PACF)")
    axes[0, 1].set_xlabel("Lags")
    
    # Exogenous Correlation Chart (CCF)
    ccf_vix = [train["de_trended_ret"].corr(train["VIX_ret"].shift(lag)) for lag in range(11)]
    axes[1, 0].stem(range(11), ccf_vix)
    axes[1, 0].set_title("3. CCF: VIX Return Lag -> De-trended Return")
    axes[1, 0].set_xlabel("Lag (Days)")
    axes[1, 0].set_ylabel("Korelasi")
    
    # Distribution Histogram
    axes[1, 1].hist(train["de_trended_ret"], bins=100, color="green", alpha=0.7, density=True)
    axes[1, 1].set_title("5. Histogram & Distribusi De-trended Return")
    
    plt.tight_layout()
    plt.savefig("detrended_diagnostics_plot.png", dpi=150)
    plt.close()
    print("\nVisualisasi diagnostik disimpan ke 'detrended_diagnostics_plot.png'")

if __name__ == "__main__":
    main()
