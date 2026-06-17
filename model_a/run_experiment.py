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
SCENARIOS = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}


def load_best_params(scenario_name):
    best_path = RESULTS_DIR / "best_params.json"
    if not best_path.exists():
        return {}
    with open(best_path) as f:
        data = json.load(f)
    entry = data.get(scenario_name, {})
    return entry.get("params", {})


def run_scenario(scenario_name, train_ratio):
    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    df = load_train()
    train_df, test_df = temporal_split(df, train_ratio)
    n_train_raw = len(train_df)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    X_full, y_full = build_features(full_df)

    n_nan = len(full_df) - len(X_full)
    n_train_feat = n_train_raw - n_nan

    X_train, y_train = X_full.iloc[:n_train_feat], y_full.iloc[:n_train_feat]
    X_test, y_test = X_full.iloc[n_train_feat:], y_full.iloc[n_train_feat:]

    print(f"  Train: {len(train_df)} -> {len(X_train)} rows (dropped {n_nan} NaN)")
    print(f"  Test: {len(test_df)} -> {len(X_test)} rows")
    print(f"  Features: {X_train.shape[1]}")

    best = load_best_params(scenario_name)
    alpha = best.get("alpha", 1.0)
    l1_ratio = best.get("l1_ratio", 0.5)
    print(f"  Params: alpha={alpha:.6f}, l1_ratio={l1_ratio:.4f}")

    model = ElasticNetPAC(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_all(y_test, y_pred)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    test_dates = test_df["Date"].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "date": test_dates,
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    pred_path = RESULTS_DIR / f"predictions_{scenario_name.replace('/', '_')}.csv"
    pred_df.to_csv(pred_path, index=False)

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

    return {"scenario": scenario_name, "train_ratio": train_ratio, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Run Model A experiments")
    parser.add_argument("--scenario", type=str, default=None)
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
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()