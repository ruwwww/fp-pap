import sys
import json
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, load_test, temporal_split, evaluate_all, plot_actual_vs_predicted
from model_c.model import RidgeRFEnsemble
from model_c.features import build_features_ridge, build_features_rf

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
    Xr_full, y_full = build_features_ridge(full_df)
    Xrf_full, _ = build_features_rf(full_df)

    n_nan_ridge = len(full_df) - len(Xr_full)
    n_nan_rf = len(full_df) - len(Xrf_full)

    n_train_ridge = n_train_raw - n_nan_ridge
    n_train_rf = n_train_raw - n_nan_rf

    Xr_train, y_train_r = Xr_full.iloc[:n_train_ridge], y_full.iloc[:n_train_ridge]
    Xr_test = Xr_full.iloc[n_train_ridge:]
    Xrf_train = Xrf_full.iloc[:n_train_rf]
    Xrf_test = Xrf_full.iloc[n_train_rf:]

    y_test_r = y_full.iloc[n_train_ridge:]

    print(f"  Ridge: train={len(Xr_train)}, test={len(Xr_test)}, features={Xr_train.shape[1]}")
    print(f"  RF:    train={len(Xrf_train)}, test={len(Xrf_test)}, features={Xrf_train.shape[1]}")

    min_train = min(len(Xr_train), len(Xrf_train))
    min_test = min(len(Xr_test), len(Xrf_test))

    model = RidgeRFEnsemble(w_ridge=0.7, w_rf=0.3)
    model.fit(Xr_train.iloc[:min_train], Xrf_train.iloc[:min_train], y_train_r.iloc[:min_train])

    y_pred = model.predict(Xr_test.iloc[:min_test], Xrf_test.iloc[:min_test])
    y_test = y_test_r.iloc[:min_test]

    metrics = evaluate_all(y_test, y_pred)
    print(f"\n  Results:")
    for k, v in metrics.items():
        print(f"    {k:>6}: {v:.4f}")

    metrics_path = RESULTS_DIR / f"metrics_{scenario_name.replace('/', '_')}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    test_dates = test_df["Date"].reset_index(drop=True).iloc[:min_test]
    pred_df = pd.DataFrame({
        "date": test_dates.values,
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
        model_name="RidgeRF",
        scenario=scenario_name,
        save_path=str(plot_path),
    )

    return {"scenario": scenario_name, "train_ratio": train_ratio, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Run Model C experiments")
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