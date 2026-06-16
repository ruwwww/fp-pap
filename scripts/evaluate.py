"""
scripts/evaluate.py
────────────────────
Reads results_summary.csv and generates the full evaluation report
(Markdown tables + plots). Run after train.py.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --results results/metrics/results_summary.csv
  python scripts/evaluate.py --report-dir report/
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.plots import plot_metrics_comparison, plot_heatmap, plot_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("evaluate")


def generate_markdown_table(df: pd.DataFrame, scenario: str, decimals: int = 4) -> str:
    """Generate a Markdown table for one scenario."""
    sub = df[df["Scenario"] == scenario].copy()
    sub = sub.sort_values("RMSE")

    cols = ["Model", "Category", "RMSE", "MAPE (%)", "MAE", "R²", "Fit Time (s)"]
    cols = [c for c in cols if c in sub.columns]

    lines = []
    # Header
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for _, row in sub.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.{decimals}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_full_report(df: pd.DataFrame, plots_dir: Path, report_dir: Path) -> str:
    """Build complete Markdown evaluation report."""
    lines = []
    lines.append("# Rekapitulasi Hasil Forecasting\n")
    lines.append(f"*Generated automatically from `results/metrics/results_summary.csv`*\n")

    # ── Overview ─────────────────────────────────────────────────────
    lines.append("## Overview\n")
    n_models    = df["Model"].nunique()
    n_scenarios = df["Scenario"].nunique()
    lines.append(f"- **Total models evaluated**: {n_models}")
    lines.append(f"- **Scenarios**: {n_scenarios} (80/20, 70/30, 60/40)")
    lines.append(f"- **Metrics**: RMSE, MAPE, MAE, R²")
    lines.append("")

    # ── Per-Scenario Tables ───────────────────────────────────────────
    lines.append("## Hasil Per Skenario\n")
    for scenario in sorted(df["Scenario"].unique()):
        lines.append(f"### {scenario}\n")
        lines.append(generate_markdown_table(df, scenario))
        lines.append("")

        # Best model for this scenario
        best = df[df["Scenario"] == scenario].sort_values("RMSE").iloc[0]
        lines.append(f"> **Best model**: `{best['Model']}` — "
                     f"RMSE={best['RMSE']:.4f}, MAPE={best['MAPE (%)']:.2f}%, "
                     f"MAE={best['MAE']:.4f}, R²={best['R²']:.4f}\n")

    # ── Best per scenario summary ─────────────────────────────────────
    lines.append("## Best Model Per Scenario\n")
    best_rows = df.loc[df.groupby("Scenario")["RMSE"].idxmin()]
    lines.append(generate_markdown_table(best_rows.assign(Scenario=best_rows["Scenario"]), None) if False else "")
    # Manual table
    lines.append("| Scenario | Best Model | RMSE | MAPE (%) | MAE | R² |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in best_rows.sort_values("Scenario").iterrows():
        lines.append(
            f"| {row['Scenario']} | **{row['Model']}** | {row['RMSE']:.4f} | "
            f"{row['MAPE (%)']:.2f} | {row['MAE']:.4f} | {row['R²']:.4f} |"
        )
    lines.append("")

    # ── Category Analysis ─────────────────────────────────────────────
    lines.append("## Analisis Per Kategori Model\n")
    cat_stats = df.groupby("Category")[["RMSE", "MAPE (%)", "MAE", "R²"]].mean().round(4)
    lines.append("**Rata-rata metrik per kategori:**\n")
    lines.append("| Category | RMSE | MAPE (%) | MAE | R² |")
    lines.append("|---|---|---|---|---|")
    for cat, row in cat_stats.iterrows():
        lines.append(f"| {cat} | {row['RMSE']:.4f} | {row['MAPE (%)']:.2f} | {row['MAE']:.4f} | {row['R²']:.4f} |")
    lines.append("")

    # ── Plot references ───────────────────────────────────────────────
    lines.append("## Visualisasi\n")
    lines.append("### Heatmap Metrik (Model × Skenario)\n")
    for metric in ["RMSE", "MAPE (%)", "MAE", "R²"]:
        safe = metric.replace(" ", "_").replace("²","2").replace("(","").replace(")","")
        img_path = plots_dir / f"heatmap_{safe}.png"
        if img_path.exists():
            rel = os.path.relpath(str(img_path), str(report_dir))
            lines.append(f"![Heatmap {metric}]({rel})\n")

    lines.append("### Actual vs Predicted Plots\n")
    for img in sorted(plots_dir.glob("*.png")):
        if "heatmap" in img.name or "table" in img.name or "forecast" in img.name:
            continue
        rel = os.path.relpath(str(img), str(report_dir))
        lines.append(f"![{img.stem}]({rel})\n")

    lines.append("### Forecast Plot\n")
    for img in sorted(plots_dir.glob("forecast_*.png")):
        rel = os.path.relpath(str(img), str(report_dir))
        lines.append(f"![{img.stem}]({rel})\n")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="EAS — Evaluation Report Generator")
    p.add_argument("--results",    default="results/metrics/results_summary.csv")
    p.add_argument("--plots-dir",  default="results/plots")
    p.add_argument("--report-dir", default="report")
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        log.error(f"Results file not found: {results_path}")
        log.error("Run scripts/train.py first.")
        sys.exit(1)

    df = pd.read_csv(results_path)
    plots_dir  = Path(args.plots_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)

    log.info(f"Loaded results: {df.shape} ({df['Model'].nunique()} models, {df['Scenario'].nunique()} scenarios)")

    # ── Generate extra comparison plots ──────────────────────────────
    for scenario in df["Scenario"].unique():
        for metric in ["RMSE", "MAPE (%)", "MAE", "R²"]:
            safe = scenario.replace("/", "_").replace(" ", "_")
            metric_safe = metric.replace(" ", "_").replace("²","2").replace("(","").replace(")","")
            try:
                fig = plot_metrics_comparison(
                    df, metric=metric, scenario=scenario,
                    save_path=str(plots_dir / f"compare_{metric_safe}_{safe}.png"),
                )
                plt.close("all")
            except Exception as e:
                log.debug(f"Comparison plot failed: {e}")

    # ── Generate full Markdown report ─────────────────────────────────
    report_md = generate_full_report(df, plots_dir, report_dir)
    report_path = report_dir / "02_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    log.info(f"Report generated: {report_path}")

    # ── CSV export per scenario ───────────────────────────────────────
    for scenario in df["Scenario"].unique():
        safe = scenario.replace("/", "_").replace(" ", "_")
        path = report_dir / "tables" / f"results_{safe}.csv"
        path.parent.mkdir(exist_ok=True)
        df[df["Scenario"] == scenario].to_csv(path, index=False)
    log.info("Per-scenario CSVs saved to report/tables/")

    log.info("\n--- Summary ---")
    best = df.loc[df.groupby("Scenario")["RMSE"].idxmin()]
    print(best[["Scenario", "Model", "RMSE", "MAPE (%)", "MAE", "R²"]].to_string(index=False))


if __name__ == "__main__":
    main()