"""
scripts/run_all.py
───────────────────
Full pipeline orchestrator. Runs all stages sequentially or selectively.

Usage:
  # Full pipeline
  python scripts/run_all.py --data data/raw/train.csv

  # Skip search (use default params)
  python scripts/run_all.py --data data/raw/train.csv --skip-search

  # Skip search + skip forecast (train + evaluate only)
  python scripts/run_all.py --data data/raw/train.csv --skip-search --skip-forecast

  # Parallel ML, 50 search trials, then forecast with best model
  python scripts/run_all.py --data data/raw/train.csv --parallel --workers 4 --trials 50

  # Quick smoke run (ML only, 80/20, 10 search trials)
  python scripts/run_all.py --data data/raw/train.csv --skip-category DL \\
      --scenarios 80_20 --trials 10 --skip-forecast

  # Background-friendly (redirect stdout+stderr)
  python scripts/run_all.py --data data/raw/train.csv > logs/run.log 2>&1
"""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/run_all.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("run_all")


def run_stage(cmd: list, stage_name: str) -> bool:
    """Run a subprocess stage and return True if success."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: {stage_name}")
    log.info(f"CMD  : {' '.join(cmd)}")
    log.info(f"{'='*60}")
    t0 = time.time()

    result = subprocess.run(
        cmd,
        capture_output=False,   # show output in real-time
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error(f"STAGE FAILED: {stage_name} (exit={result.returncode}) in {elapsed:.1f}s")
        return False
    else:
        log.info(f"STAGE OK: {stage_name} in {elapsed:.1f}s")
        return True


def main():
    p = argparse.ArgumentParser(description="EAS — Full Pipeline Orchestrator")
    # Data
    p.add_argument("--data",      default="data/raw/data_train.csv")
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--models-config", default="config/models_config.yaml")
    # Training
    p.add_argument("--models",    default=None, help="Comma-separated model keys")
    p.add_argument("--skip",      default=None, help="Comma-separated models to skip")
    p.add_argument("--skip-category", default=None, help="ML or DL")
    p.add_argument("--scenarios", default=None, help="e.g. 80_20,70_30,60_40")
    p.add_argument("--parallel",  action="store_true", help="Parallel ML training")
    p.add_argument("--workers",   type=int, default=4)
    # Search
    p.add_argument("--skip-search",   action="store_true", help="Skip hyperparameter search")
    p.add_argument("--trials",        type=int, default=50)
    p.add_argument("--search-models", default=None, help="Models to search (default: all ML enabled)")
    p.add_argument("--search-jobs",   type=int, default=1)
    # Forecast
    p.add_argument("--skip-forecast", action="store_true")
    p.add_argument("--forecast-model", default=None, help="Model to use for forecast")
    p.add_argument("--forecast-start", default="2023-06-01")
    p.add_argument("--forecast-end",   default="2026-05-29")
    p.add_argument("--freq",      default=None)
    # Report
    p.add_argument("--report-dir", default="report")
    args = p.parse_args()

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(exist_ok=True)
    Path("results/plots").mkdir(exist_ok=True)
    Path("results/search").mkdir(exist_ok=True)
    Path("data/submissions").mkdir(parents=True, exist_ok=True)

    python = sys.executable
    all_ok = True
    t_start = time.time()

    # ── STAGE 1: Hyperparameter Search ───────────────────────────────
    if not args.skip_search:
        search_cmd = [
            python, "scripts/search.py",
            "--trials", str(args.trials),
            "--data", args.data,
            "--config", args.config,
            "--models-config", args.models_config,
            "--jobs", str(args.search_jobs),
        ]
        if args.search_models:
            search_cmd += ["--models", args.search_models]
        elif args.skip_category:
            # Search opposite category only
            cat = "DL" if args.skip_category == "ML" else "ML"
            search_cmd += ["--category", cat]
        else:
            search_cmd += ["--category", "ML"]   # default: search ML only (faster)

        ok = run_stage(search_cmd, "Hyperparameter Search")
        all_ok = all_ok and ok
    else:
        log.info("Skipping hyperparameter search.")

    # ── STAGE 2: Training ─────────────────────────────────────────────
    train_cmd = [
        python, "scripts/train.py",
        "--data", args.data,
        "--config", args.config,
        "--models-config", args.models_config,
    ]
    if not args.skip_search:
        train_cmd.append("--use-best-params")
    if args.parallel:
        train_cmd += ["--parallel", "--workers", str(args.workers)]
    if args.models:
        train_cmd += ["--models", args.models]
    if args.skip:
        train_cmd += ["--skip", args.skip]
    if args.skip_category:
        train_cmd += ["--skip-category", args.skip_category]
    if args.scenarios:
        train_cmd += ["--scenarios", args.scenarios]

    ok = run_stage(train_cmd, "Model Training")
    all_ok = all_ok and ok

    # ── STAGE 3: Evaluate & Report ────────────────────────────────────
    eval_cmd = [
        python, "scripts/evaluate.py",
        "--report-dir", args.report_dir,
    ]
    ok = run_stage(eval_cmd, "Evaluation & Report Generation")
    all_ok = all_ok and ok

    # ── STAGE 4: Forecast ─────────────────────────────────────────────
    if not args.skip_forecast:
        # Auto-pick best model if not specified
        forecast_model = args.forecast_model
        if not forecast_model:
            import pandas as pd
            try:
                results_df = pd.read_csv("results/metrics/results_summary.csv")
                best = results_df.sort_values("RMSE").iloc[0]
                # Map model name to key — attempt lowercase
                forecast_model = best["Model"].lower().replace("-", "_").replace(" ", "_")
                log.info(f"Auto-selected best model for forecast: {forecast_model}")
            except Exception:
                forecast_model = "xgboost"

        forecast_cmd = [
            python, "scripts/forecast.py",
            "--data", args.data,
            "--config", args.config,
            "--models-config", args.models_config,
            "--model", forecast_model,
            "--start", args.forecast_start,
            "--end", args.forecast_end,
            "--out", "data/submissions/forecast.csv",
        ]
        if not args.skip_search:
            forecast_cmd.append("--use-best-params")
        if args.freq:
            forecast_cmd += ["--freq", args.freq]

        ok = run_stage(forecast_cmd, "Future Forecasting")
        all_ok = all_ok and ok
    else:
        log.info("Skipping forecast.")

    # ── Summary ───────────────────────────────────────────────────────
    total = time.time() - t_start
    log.info(f"\n{'='*60}")
    log.info(f"PIPELINE {'COMPLETE' if all_ok else 'FINISHED WITH ERRORS'}")
    log.info(f"Total time: {total/60:.1f} min")
    log.info(f"Results   : results/metrics/results_summary.csv")
    log.info(f"Report    : {args.report_dir}/02_results.md")
    log.info(f"Forecast  : data/submissions/forecast.csv")
    log.info(f"Plots     : results/plots/")
    log.info(f"{'='*60}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
