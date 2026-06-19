#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"
TEST_EXOG_CSV = ROOT / "data_test.csv"
TEST_ACTUAL_CSV = ROOT / "data_test_actual.csv"

def main():
    # 1. Load Train set
    train = pd.read_csv(TRAIN_CSV)
    train["Date"] = pd.to_datetime(train["Date"])
    train = train.sort_values("Date").reset_index(drop=True)
    
    # 2. Load Test Exogenous and Actual sets
    test_exog = pd.read_csv(TEST_EXOG_CSV)
    test_actual = pd.read_csv(TEST_ACTUAL_CSV)
    
    # Combine test set columns
    test = test_exog.copy()
    test["USDIDR"] = test_actual["USDIDR"]
    test["Date"] = pd.to_datetime(test["Date"])
    test = test.sort_values("Date").reset_index(drop=True)
    
    # 3. Concatenate Train and Test sets to get the FULL timeline
    full_df = pd.concat([train, test], ignore_index=True)
    full_df = full_df.sort_values("Date").reset_index(drop=True)
    
    # Numerical feature columns to plot
    feature_cols = [col for col in train.columns if col != "Date"]
    
    # Set up plotting grid
    n_cols = len(feature_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3 * n_cols), sharex=True)
    
    # Find split date index
    split_date = train["Date"].max()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        
        # Plot train portion in blue, test portion in red to clearly show regime boundaries
        train_part = full_df[full_df["Date"] <= split_date]
        test_part = full_df[full_df["Date"] > split_date]
        
        ax.plot(train_part["Date"], train_part[col], color="royalblue", label=f"Train {col}")
        ax.plot(test_part["Date"], test_part[col], color="firebrick", label=f"Test {col} (OOS)")
        
        # Draw vertical line separating train and test
        ax.axvline(x=split_date, color="black", linestyle="--", alpha=0.8, label="Train/Test Split")
        
        ax.set_title(f"Full Timeline Feature: {col}", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.savefig("full_features_check.png", dpi=150)
    plt.close()
    print("Full timeline visualization plot saved successfully as 'full_features_check.png'")

if __name__ == "__main__":
    main()
