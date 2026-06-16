"""
run_benchmark.py
─────────────────
CLI entrypoint to run the full benchmark pipeline from terminal:

    python run_benchmark.py --data data/raw/train.csv --target value --date date

Options:
  --data       Path to CSV dataset
  --target     Name of target column
  --date       Name of date column
  --skip       Comma-separated model names to skip
  --only       Comma-separated model names to run exclusively
  --no-mlflow  Disable MLflow tracking
"""
import argparse
import logging
import sys
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="EAS Time Series Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--date", default=None, help="Date column name")
    parser.add_argument("--skip", default="", help="Comma-separated models to skip")
    parser.add_argument("--only", default="", help="Comma-separated models to run only")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow")
    parser.add_argument("--config", default="config/config.yaml", help="Config path")
    parser.add_argument("--models-config", default="config/models_config.yaml")
    args = parser.parse_args()

    # ── Load config and override if CLI args provided ─────────────────
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.target:
        cfg["project"]["target_column"] = args.target
    if args.date:
        cfg["project"]["date_column"] = args.date

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\n📂 Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    date_col = cfg["project"]["date_column"]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    print(f"   Shape: {df.shape} | Columns: {list(df.columns)}")

    # ── Run benchmark ─────────────────────────────────────────────────
    from benchmarking import BenchmarkRunner
    runner = BenchmarkRunner(
        config_path=args.config,
        models_config_path=args.models_config,
        use_mlflow=not args.no_mlflow,
    )
    # Override config with CLI
    runner.target_col = cfg["project"]["target_column"]
    runner.date_col = cfg["project"]["date_column"]

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None

    results = runner.run(df, skip_models=skip, only_models=only)
    runner.save_results()

    print("\n🏆 Best models per scenario:")
    print(runner.best_per_scenario().to_string(index=False))


if __name__ == "__main__":
    main()
