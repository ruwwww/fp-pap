import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, temporal_split, evaluate_all, plot_actual_vs_predicted
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

    train_feat = prepare_raw_features(train_df)
    test_feat = prepare_raw_features(test_df)

    has_target = "USDIDR" in train_feat.columns
    feature_cols = [c for c in train_feat.columns if c != "USDIDR"]
    y_train = train_feat["USDIDR"].values if has_target else None
    y_test = test_feat["USDIDR"].values if has_target else None
    X_train = train_feat[feature_cols].values
    X_test = test_feat[feature_cols].values

    print(f"  Raw features: {len(feature_cols)}")
    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

    lookback = 30
    model = GRUModel(lookback=lookback, hidden_size=32, dropout=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_dates = test_df["Date"].iloc[lookback:].reset_index(drop=True)
    y_test_aligned = y_test[lookback:]
    y_pred_aligned = y_pred

    min_len = min(len(y_test_aligned), len(y_pred_aligned), len(test_dates))
    y_test_aligned = y_test_aligned[:min_len]
    y_pred_aligned = y_pred_aligned[:min_len]
    test_dates = test_dates.iloc[:min_len]

    metrics = evaluate_all(y_test_aligned, y_pred_aligned)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "date": test_dates,
        "y_true": y_test_aligned,
        "y_pred": y_pred_aligned,
    })
    pred_path = RESULTS_DIR / f"predictions_{scenario_name.replace('/', '_')}.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_path = RESULTS_DIR / f"actual_vs_predicted_{scenario_name.replace('/', '_')}.png"
    plot_actual_vs_predicted(
        train_dates=train_df["Date"],
        train_actual=train_df["USDIDR"],
        test_dates=test_dates,
        test_actual=y_test_aligned,
        test_pred=y_pred_aligned,
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