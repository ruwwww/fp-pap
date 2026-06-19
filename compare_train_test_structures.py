#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    train = pd.read_csv("data_train.csv")
    test_exog = pd.read_csv("data_test.csv")
    test_actual = pd.read_csv("data_test_actual.csv")
    
    # Analyze macro features standard deviation between train and test OOS periods
    print("=== INVESTIGASI STRUKTUR VOLATILITAS MAKRO (TRAIN VS TEST) ===")
    
    combined = pd.concat([train, test_exog], ignore_index=True)
    for col in ["SP500", "GOLD", "OIL", "IHSG", "VIX"]:
        combined[f"{col}_ret"] = np.log(combined[col]).diff().fillna(0.0)
    combined["bi_rate_change"] = combined["BI_rate"].diff().fillna(0.0)
    
    # Split back
    train_processed = combined.iloc[:len(train)]
    test_processed = combined.iloc[len(train):]
    
    print("\n--- Standard Deviation of Returns (Volatilitas Harian) ---")
    for col in ["SP500_ret", "VIX_ret", "bi_rate_change"]:
        std_train = train_processed[col].std()
        std_test = test_processed[col].std()
        ratio = std_test / std_train if std_train > 0 else 0
        print(f"{col:<20} | Train Vol: {std_train:.6f} | Test Vol: {std_test:.6f} | Ratio (Test/Train): {ratio:.2f}x")
        
    print("\n--- Rata-rata Perubahan Arah (Mean Return) ---")
    for col in ["SP500_ret", "VIX_ret", "bi_rate_change"]:
        mean_train = train_processed[col].mean()
        mean_test = test_processed[col].mean()
        print(f"{col:<20} | Train Mean: {mean_train:.6f} | Test Mean: {mean_test:.6f}")
        
    # Correlation analysis inside Test OOS period
    test_actual["USDIDR_ret"] = np.log(test_actual["USDIDR"]).diff().fillna(0.0)
    test_processed = test_processed.reset_index(drop=True)
    
    print("\n--- Korelasi Aktual OOS Return dengan Exogenous Lags ---")
    for col in ["SP500_ret", "VIX_ret", "bi_rate_change"]:
        corr = test_actual["USDIDR_ret"].corr(test_processed[col].shift(1))
        print(f"Korelasi USDIDR_ret dengan {col}_lag1 di OOS: {corr:.4f}")

if __name__ == "__main__":
    main()
