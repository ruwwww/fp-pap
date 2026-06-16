import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, temporal_split, evaluate_all, plot_actual_vs_predicted
from model_d.model import HybridRidgeGRU
from model_d.features import build_features_ridge, prepare_gru_raw

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCENARIOS = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}


def run_scenario(scenario_name, train_ratio):
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    df = load_train()
    train_df, test_df = temporal_split(df, train_ratio)

    Xr_train, y_train = build_features_ridge(train_df)
    Xr_test, y_test = build_features_ridge(test_df)
    seq_train = prepare_gru_raw(train_df)
    seq_test = prepare_gru_raw(test_df)

    has_target = "USDIDR" in seq_train.columns
    seq_cols = [c for c in seq_train.columns if c != "USDIDR"]
    y_seq_train = seq_train["USDIDR"].values if has_target else None
    y_seq_test = seq_test["USDIDR"].values if has_target else None
    X_seq_train = seq_train[seq_cols].values
    X_seq_test = seq_test[seq_cols].values

    lookback = 30

    min_train = min(len(Xr_train), len(X_seq_train))
    Xr_train, X_seq_train, y_train = Xr_train.iloc[:min_train], X_seq_train[:min_train], y_train.iloc[:min_train]
    X_seq_train = np.asarray(X_seq_train, dtype=float)

    min_test = min(len(Xr_test), len(X_seq_test))
    Xr_test = Xr_test.iloc[:min_test]
    X_seq_test = X_seq_test[:min_test]
    y_test_aligned = y_test.iloc[:min_test] if hasattr(y_test, 'iloc') else y_test[:min_test]

    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")
    print(f"  Ridge features: {Xr_train.shape[1]}, GRU features: {X_seq_train.shape[1]}")

    model = HybridRidgeGRU(lookback=lookback, hidden_size=32, dropout=0.1)
    model.fit(Xr_train, X_seq_train, y_train.values if hasattr(y_train, 'values') else y_train)

    y_pred = model.predict(Xr_test, X_seq_test)

    test_dates = test_df["Date"].iloc[lookback:].reset_index(drop=True)
    y_test_arr = np.asarray(y_test_aligned)[lookback:]
    y_pred_aligned = y_pred[:len(y_test_arr)]

    min_len = min(len(y_test_arr), len(y_pred_aligned), len(test_dates))
    y_test_arr = y_test_arr[:min_len]
    y_pred_aligned = y_pred_aligned[:min_len]
    test_dates = test_dates.iloc[:min_len]

    metrics = evaluate_all(y_test_arr, y_pred_aligned)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "date": test_dates,
        "y_true": y_test_arr,
        "y_pred": y_pred_aligned,
    })
    pred_path = RESULTS_DIR / f"predictions_{scenario_name.replace('/', '_')}.csv"
    pred_df.to_csv(pred_path, index=False)

    plot_path = RESULTS_DIR / f"actual_vs_predicted_{scenario_name.replace('/', '_')}.png"
    plot_actual_vs_predicted(
        train_dates=train_df["Date"],
        train_actual=train_df["USDIDR"],
        test_dates=test_dates,
        test_actual=y_test_arr,
        test_pred=y_pred_aligned,
        model_name="HybridRidgeGRU",
        scenario=scenario_name,
        save_path=str(plot_path),
    )

    return {"scenario": scenario_name, "train_ratio": train_ratio, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Run Model D experiments")
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