import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, load_test, temporal_split, evaluate_all, plot_actual_vs_predicted
from model_b.model import GRUModel
from model_b.features import prepare_raw_features

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCENARIOS = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}


def run_scenario(scenario_name, train_ratio):
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    df = load_train()
    train_df, test_df = temporal_split(df, train_ratio)
    n_train_raw = len(train_df)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_feat = prepare_raw_features(full_df)

    n_nan = len(full_df) - len(full_feat)
    n_train_feat = n_train_raw - n_nan

    feature_cols = [c for c in full_feat.columns if c != "USDIDR"]
    y_full = full_feat["USDIDR"].values
    X_full = full_feat[feature_cols].values

    X_train = X_full[:n_train_feat]
    y_train = y_full[:n_train_feat]
    X_test = X_full[n_train_feat:]
    y_test = y_full[n_train_feat:]

    lookback = 30

    print(f"  Train: {len(train_df)} -> {len(X_train)} rows")
    print(f"  Test: {len(test_df)} -> {len(X_test)} rows")
    print(f"  Raw features: {len(feature_cols)}")

    model = GRUModel(lookback=lookback, hidden_size=32, dropout=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_dates = test_df["Date"].iloc[lookback:].reset_index(drop=True)
    y_test_arr = y_test[lookback:]
    min_len = min(len(y_test_arr), len(y_pred), len(test_dates))
    y_test_arr = y_test_arr[:min_len]
    y_pred = y_pred[:min_len]
    test_dates = test_dates.iloc[:min_len]

    metrics = evaluate_all(y_test_arr, y_pred)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "date": test_dates,
        "y_true": y_test_arr,
        "y_pred": y_pred,
    })
    pred_path = RESULTS_DIR / f"predictions_{scenario_name.replace('/', '_')}.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_path = RESULTS_DIR / f"actual_vs_predicted_{scenario_name.replace('/', '_')}.png"
    plot_actual_vs_predicted(
        train_dates=train_df["Date"],
        train_actual=train_df["USDIDR"],
        test_dates=test_dates,
        test_actual=y_test_arr,
        test_pred=y_pred,
        model_name="GRU",
        scenario=scenario_name,
        save_path=str(plot_path),
    )

    return {"scenario": scenario_name, "train_ratio": train_ratio, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Run Model B experiments")
    parser.add_argument("--scenario", type=str, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    all_results = []
    for name, ratio in scenarios.items():
        result = run_scenario(name, ratio)
        all_results.append(result)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()