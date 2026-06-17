import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, temporal_split, rmse
from model_a.model import ElasticNetPAC
from model_a.features import build_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCENARIOS = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}


def prepare_data(train_ratio):
    df = load_train()
    train_df, test_df = temporal_split(df, train_ratio)
    n_train_raw = len(train_df)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    X_full, y_full = build_features(full_df)

    n_nan = len(full_df) - len(X_full)
    n_train_feat = n_train_raw - n_nan
    n_val = max(1, int(n_train_feat * 0.2))

    X_train_full = X_full.iloc[:n_train_feat]
    y_train_full = y_full.iloc[:n_train_feat]

    X_tr = X_train_full.iloc[:-n_val]
    y_tr = y_train_full.iloc[:-n_val]
    X_val = X_train_full.iloc[-n_val:]
    y_val = y_train_full.iloc[-n_val:]

    return X_tr, y_tr, X_val, y_val


def objective(trial, X_tr, y_tr, X_val, y_val):
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    model = ElasticNetPAC(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)

    return rmse(y_val, y_pred)


def run_search(scenario_name, train_ratio, n_trials):
    print(f"\n{'='*60}")
    print(f"  Hyperparameter Search — {scenario_name}")
    print(f"{'='*60}")

    X_tr, y_tr, X_val, y_val = prepare_data(train_ratio)
    print(f"  Train: {len(X_tr)}, Validation: {len(X_val)}, Features: {X_tr.shape[1]}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_val, y_val), n_trials=n_trials)

    best = study.best_params
    best_val_rmse = study.best_value

    print(f"\n  Best val RMSE: {best_val_rmse:.4f}")
    print(f"  Best params:   alpha={best['alpha']:.6f}, l1_ratio={best['l1_ratio']:.4f}")

    return best, best_val_rmse


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for Model A")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            sys.exit(1)
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    all_best = {}
    for name, ratio in scenarios.items():
        best_params, best_rmse = run_search(name, ratio, args.trials)
        all_best[name] = {"params": best_params, "val_rmse": best_rmse}

    best_path = RESULTS_DIR / "best_params.json"
    with open(best_path, "w") as f:
        json.dump(all_best, f, indent=2)

    print(f"\n{'='*60}")
    print("  Best Params Summary")
    print(f"{'='*60}")
    for scenario, info in all_best.items():
        p = info["params"]
        print(f"  {scenario}: alpha={p['alpha']:.6f}  l1_ratio={p['l1_ratio']:.4f}  (val RMSE={info['val_rmse']:.4f})")
    print(f"\n  Saved to {best_path}")


if __name__ == "__main__":
    main()