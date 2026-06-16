"""
src/evaluation/metrics.py
──────────────────────────
Evaluation metrics for time series forecasting:
  - RMSE, MAPE, MAE, R² (sesuai instruksi)
  - Comprehensive result table builder
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Individual metrics ─────────────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-Squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE (bonus metric)."""
    return float(
        np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    )


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics at once.

    Returns:
        dict with RMSE, MAPE, MAE, R2, SMAPE
    """
    y_true = np.array(y_true, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    return {
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }


# ── Results table builder ──────────────────────────────────────────────────
class ResultsTable:
    """
    Accumulate results across models and scenarios,
    then produce a formatted summary DataFrame.

    Usage:
        table = ResultsTable()
        table.add(model_name="XGBoost", category="ML",
                  scenario="80/20", y_true=..., y_pred=...)
        df = table.to_dataframe()
        table.print_summary()
    """

    def __init__(self):
        self._records: List[Dict] = []

    def add(
        self,
        model_name: str,
        category: str,
        scenario: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fit_time: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Add a result entry and return the computed metrics."""
        metrics = evaluate_all(y_true, y_pred)
        record = {
            "Model": model_name,
            "Category": category,
            "Scenario": scenario,
            "RMSE": metrics["RMSE"],
            "MAPE (%)": metrics["MAPE"],
            "MAE": metrics["MAE"],
            "R²": metrics["R2"],
            "SMAPE (%)": metrics["SMAPE"],
            "Fit Time (s)": round(fit_time, 2),
        }
        if extra:
            record.update(extra)
        self._records.append(record)
        logger.info(
            f"  [{scenario}] {model_name:20s} | "
            f"RMSE={metrics['RMSE']:.4f} | MAPE={metrics['MAPE']:.2f}% | "
            f"MAE={metrics['MAE']:.4f} | R²={metrics['R2']:.4f}"
        )
        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Return all results as a sorted DataFrame."""
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        return df.sort_values(["Scenario", "RMSE"]).reset_index(drop=True)

    def print_summary(self, top_n: int = 5) -> None:
        """Print a rich formatted table to terminal."""
        try:
            from rich.console import Console
            from rich.table import Table as RichTable

            console = Console()
            df = self.to_dataframe()

            for scenario in df["Scenario"].unique():
                sub = df[df["Scenario"] == scenario].head(top_n)
                table = RichTable(title=f"📊 Results — {scenario}", show_lines=True)
                for col in sub.columns:
                    table.add_column(col, justify="right" if col not in ("Model", "Category", "Scenario") else "left")
                for _, row in sub.iterrows():
                    vals = []
                    for col in sub.columns:
                        v = row[col]
                        if isinstance(v, float):
                            vals.append(f"{v:.4f}")
                        else:
                            vals.append(str(v))
                    table.add_row(*vals)
                console.print(table)
        except ImportError:
            print(self.to_dataframe().to_string())

    def save(self, path: str = "results/metrics/results_summary.csv") -> None:
        """Save results table to CSV."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")

    def best_model(self, scenario: str, metric: str = "RMSE") -> Dict:
        """Return the best model for a given scenario."""
        df = self.to_dataframe()
        sub = df[df["Scenario"] == scenario]
        ascending = metric != "R²"
        return sub.sort_values(metric, ascending=ascending).iloc[0].to_dict()
