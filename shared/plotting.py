import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional


COLORS = {"actual": "#2196F3", "train": "#4CAF50", "test": "#FF9800", "forecast": "#E91E63"}


def _save(fig, path, dpi=150):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_actual_vs_predicted(
    train_dates, train_actual,
    test_dates, test_actual, test_pred,
    model_name, scenario="",
    save_path=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    ax = axes[0]
    ax.plot(train_dates, train_actual, color=COLORS["train"], linewidth=1.2, label="Train (Actual)", alpha=0.8)
    ax.plot(test_dates, test_actual, color=COLORS["actual"], linewidth=1.5, label="Test (Actual)", alpha=0.9)
    ax.plot(test_dates, test_pred, color=COLORS["test"], linewidth=2.0, linestyle="--", label=f"{model_name} (Predicted)")
    ax.axvline(x=test_dates.iloc[0], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Train/Test split")
    ax.set_title(f"Actual vs Predicted - {model_name} [{scenario}]", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("USDIDR", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2 = axes[1]
    residuals = np.array(test_actual) - np.array(test_pred)
    ax2.bar(range(len(residuals)), residuals, color=COLORS["forecast"], alpha=0.6, width=1.0)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_title("Residuals (Actual - Predicted)", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=10)

    fig.tight_layout(pad=2.0)
    _save(fig, save_path)
    return fig


def plot_forecast(
    train_dates, train_actual,
    forecast_dates, forecast_values,
    model_name, save_path=None,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(train_dates, train_actual, color=COLORS["actual"], linewidth=1.5, label="Historical (Actual)")
    ax.plot(forecast_dates, forecast_values, color=COLORS["forecast"], linewidth=2.0, linestyle="--", label=f"{model_name} Forecast")
    ax.axvline(x=forecast_dates.iloc[0], color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Forecast start")
    ax.set_title(f"Forecast: Jun 2023 - May 2026 - {model_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("USDIDR", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
