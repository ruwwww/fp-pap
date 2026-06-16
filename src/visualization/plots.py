"""
src/visualization/plots.py
────────────────────────────
Publication-quality plots for time series forecasting reports.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── Style configuration ────────────────────────────────────────────────────
COLORS = {
    "actual": "#2196F3",
    "train": "#4CAF50",
    "test": "#FF9800",
    "forecast": "#E91E63",
    "grid": "#E0E0E0",
    "palette": ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0",
                "#00BCD4", "#FF5722", "#607D8B"],
}

def _setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette(COLORS["palette"])


def _save(fig: plt.Figure, path: Optional[str], dpi: int = 150) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Plot saved → {path}")


# ── 1. Actual vs Predicted ─────────────────────────────────────────────────
def plot_actual_vs_predicted(
    train_dates: pd.Series,
    train_actual: np.ndarray,
    test_dates: pd.Series,
    test_actual: np.ndarray,
    test_pred: np.ndarray,
    model_name: str,
    scenario: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot actual data (train + test) alongside model predictions on test set.
    Sesuai instruksi: grafik dari data aktual sampai data testing.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    # ── Main plot ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(train_dates, train_actual, color=COLORS["train"], linewidth=1.2,
            label="Train (Actual)", alpha=0.8)
    ax.plot(test_dates, test_actual, color=COLORS["actual"], linewidth=1.5,
            label="Test (Actual)", alpha=0.9)
    ax.plot(test_dates, test_pred, color=COLORS["test"], linewidth=2.0,
            linestyle="--", label=f"{model_name} (Predicted)")
    ax.axvline(x=test_dates.iloc[0], color="gray", linestyle=":", linewidth=1.5,
               alpha=0.7, label="Train/Test split")

    ax.set_title(f"Actual vs Predicted — {model_name} [{scenario}]",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ── Residuals ───────────────────────────────────────────────────────
    ax2 = axes[1]
    residuals = np.array(test_actual) - np.array(test_pred)
    ax2.bar(range(len(residuals)), residuals, color=COLORS["forecast"], alpha=0.6, width=1.0)
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.set_title("Residuals (Actual − Predicted)", fontsize=11)
    ax2.set_ylabel("Residual", fontsize=10)

    fig.tight_layout(pad=2.0)
    _save(fig, save_path)
    return fig


# ── 2. Forecast Plot ────────────────────────────────────────────────────────
def plot_forecast(
    train_dates: pd.Series,
    train_actual: np.ndarray,
    forecast_dates: pd.Series,
    forecast_values: np.ndarray,
    model_name: str,
    confidence_interval: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show historical data + future forecast (1 Jun 2023 → 29 Mei 2026).
    Sesuai instruksi poin 6.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(train_dates, train_actual, color=COLORS["actual"], linewidth=1.5,
            label="Historical (Actual)")
    ax.plot(forecast_dates, forecast_values, color=COLORS["forecast"],
            linewidth=2.0, linestyle="--", label=f"{model_name} Forecast")

    if confidence_interval is not None:
        lower, upper = confidence_interval
        ax.fill_between(forecast_dates, lower, upper,
                        color=COLORS["forecast"], alpha=0.15, label="95% CI")

    ax.axvline(x=forecast_dates.iloc[0], color="gray", linestyle=":",
               linewidth=1.5, alpha=0.7, label="Forecast start")

    ax.set_title(f"Forecast: 1 Jun 2023 → 29 Mei 2026 — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 3. Metrics Comparison Bar Chart ────────────────────────────────────────
def plot_metrics_comparison(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    scenario: str = "80/20 Split",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing all models on a given metric for one scenario."""
    _setup_style()
    sub = results_df[results_df["Scenario"] == scenario].copy()
    sub = sub.sort_values(metric, ascending=(metric != "R²"))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [COLORS["palette"][i % len(COLORS["palette"])] for i in range(len(sub))]
    bars = ax.barh(sub["Model"], sub[metric], color=colors, edgecolor="white")

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(sub[metric]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=9)

    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f"{metric} Comparison — {scenario}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 4. All-Scenarios Heatmap ───────────────────────────────────────────────
def plot_heatmap(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of model × scenario for a given metric."""
    _setup_style()
    pivot = results_df.pivot_table(index="Model", columns="Scenario", values=metric)
    fig, ax = plt.subplots(figsize=(10, len(pivot) * 0.6 + 2))
    cmap = "RdYlGn_r" if metric != "R²" else "RdYlGn"
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap, ax=ax,
                linewidths=0.5, cbar_kws={"label": metric})
    ax.set_title(f"{metric} — Model × Scenario Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 5. Rekapitulasi summary table ──────────────────────────────────────────
def plot_results_table(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a nicely formatted summary table as a matplotlib figure."""
    _setup_style()
    display_cols = ["Model", "Category", "Scenario", "RMSE", "MAPE (%)", "MAE", "R²"]
    df_show = results_df[display_cols].copy()

    # Format floats
    for col in ["RMSE", "MAPE (%)", "MAE", "R²"]:
        df_show[col] = df_show[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(14, max(4, len(df_show) * 0.35 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header styling
    for j in range(len(df_show.columns)):
        tbl[(0, j)].set_facecolor("#1565C0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Zebra rows
    for i in range(1, len(df_show) + 1):
        for j in range(len(df_show.columns)):
            tbl[(i, j)].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")

    ax.set_title("Rekapitulasi Hasil Forecasting", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ── 6. Loss Curves (for DL models) ────────────────────────────────────────
def plot_training_history(
    history,
    model_name: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot train/val loss curves from Keras history."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric in zip(axes, ["loss", "mae"]):
        if metric in history.history:
            ax.plot(history.history[metric], label=f"Train {metric.upper()}")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f"Val {metric.upper()}")
        ax.set_title(f"{model_name} — {metric.upper()}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend()

    fig.suptitle(f"Training History — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
