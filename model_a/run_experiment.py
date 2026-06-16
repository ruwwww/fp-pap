import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, load_test, temporal_split, evaluate_all, plot_actual_vs_predicted
from model_a.model import ElasticNetPAC
from model_a.features import build_features

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCENARIOS = {
    "80/20": 0.8,
    "70/30": 0.7,
    "60/40": 0.6,
}


def run_scenario(scenario_name, train_ratio):
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    df = load_train()
    train_df, test_df = temporal_split(df, train_ratio)

    print(f"  Train size: {len(train_df)}, Test size: {len(test_df)}")

    X_train, y_train = build_features(train_df)
    X_test, y_test = build_features(test_df)

    print(f"  Features: {X_train.shape[1]}")

    model = ElasticNetPAC(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_all(y_test, y_pred)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    # Save metrics
    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    test_dates = test_df["Date"].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "date": test_dates,
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    pred_path = RESULTS_DIR / f"predictions_{scenario_name.replace('/', '_')}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Plot
    plot_path = RESULTS_DIR / f"actual_vs_predicted_{scenario_name.replace('/', '_')}.png"
    plot_actual_vs_predicted(
        train_dates=train_df["Date"],
        train_actual=train_df["USDIDR"],
        test_dates=test_dates,
        test_actual=y_test.values,
        test_pred=y_pred,
        model_name="ElasticNetPAC",
        scenario=scenario_name,
        save_path=str(plot_path),
    )

    return {
        "scenario": scenario_name,
        "train_ratio": train_ratio,
        **metrics,
        "n_features": X_train.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(description="Run Model A experiments")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario (e.g. 80/20, 70/30, 60/40)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}. Choose from {list(SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    all_results = []
    for name, ratio in scenarios.items():
        result = run_scenario(name, ratio)
        all_results.append(result)

    summary_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
