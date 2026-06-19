#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(".")
TRAIN_CSV = ROOT / "data_train.csv"

def main():
    # Load dataset
    df = pd.read_csv(TRAIN_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Identify numerical feature columns to plot
    feature_cols = [col for col in df.columns if col != "Date"]
    
    # Set up plotting grid
    n_cols = len(feature_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3 * n_cols), sharex=True)
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        # Plot raw time series
        ax.plot(df["Date"], df[col], color="royalblue", label=col)
        ax.set_title(f"Train Feature: {col}", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Simple stats annotations
        mean = df[col].mean()
        std = df[col].std()
        max_val = df[col].max()
        min_val = df[col].min()
        ax.text(0.02, 0.05, f"Mean: {mean:.2f} | Std: {std:.2f} | Range: [{min_val:.2f}, {max_val:.2f}]", 
                transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.8))
        
    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.savefig("train_features_check.png", dpi=150)
    plt.close()
    print("Feature visualization plot saved successfully as 'train_features_check.png'")

if __name__ == "__main__":
    main()
