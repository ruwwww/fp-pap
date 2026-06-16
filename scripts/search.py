"""
scripts/search.py
──────────────────
Hyperparameter search using Optuna.
Search spaces are defined in config/models_config.yaml (search_space blocks).

Usage:
  # Search one model, 50 trials, on 80/20 split
  python scripts/search.py --model xgboost --trials 50

  # Search multiple models
  python scripts/search.py --models xgboost,lightgbm --trials 100

  # Search all enabled ML models
  python scripts/search.py --category ML --trials 50

  # Search with specific scenario
  python scripts/search.py --model xgboost --trials 50 --scenario 80_20

  # Parallel Optuna trials (uses Optuna's built-in parallelism)
  python scripts/search.py --model xgboost --trials 100 --jobs 4

  # Resume a previous study (add trials to existing study)
  python scripts/search.py --model xgboost --trials 50 --resume

Best params are saved to: results/search/best_params.yaml
Study databases saved to: results/search/<model>.db (SQLite)
"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.evaluation.metrics import rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/search.log", mode="a"),
    ],
)
log = logging.getLogger("search")

SCENARIO_MAP = {
    "80_20": 0.80,
    "70_30": 0.70,
    "60_40": 0.60,
}


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    """
    Convert search_space config into Optuna trial suggestions.
    Handles int, float (with log), categorical types.
    """
    params = {}
    for param_name, spec in search_space.items():
        t = spec["type"]
        if t == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"]
            )
        elif t == "float":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"],
                log=spec.get("log", False)
            )
        elif t == "categorical":
            # Convert list-of-lists to tuple for hashability
            choices = spec["choices"]
            if any(isinstance(c, list) for c in choices):
                choice_map = {str(c): c for c in choices}
                key = trial.suggest_categorical(param_name, list(choice_map.keys()))
                params[param_name] = choice_map[key]
            else:
                params[param_name] = trial.suggest_categorical(param_name, choices)
        elif t == "bool":
            params[param_name] = trial.suggest_categorical(param_name, [True, False])
    return params


def build_objective(model_key, model_spec, X_train, y_train, X_val, y_val):
    """Factory: returns an Optuna objective function for a given model."""
    import importlib

    search_space = model_spec.get("search_space", {})
    base_params  = model_spec.get("params", {}).copy()
    class_path   = model_spec["class"]
    module_path, class_name = class_path.rsplit(".", 1)

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        searched = suggest_params(trial, search_space)

        # Merge: base params + searched (searched overrides)
        params = {**base_params, **searched}

        # Special handling: 'unit_size' maps to 'units' for DL models
        if "unit_size" in params:
            params["units"] = params.pop("unit_size")

        try:
            module = importlib.import_module(module_path)
            model = getattr(module, class_name)(params=params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = rmse(y_val.values, preds)
            return score
        except Exception as e:
            log.debug(f"  Trial failed: {e}")
            raise optuna.TrialPruned()

    return objective


def run_search(
    model_key: str,
    model_spec: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    n_jobs: int = 1,
    resume: bool = False,
    study_dir: str = "results/search",
) -> dict:
    """Run Optuna study for one model. Returns best params dict."""
    Path(study_dir).mkdir(parents=True, exist_ok=True)
    db_path = f"sqlite:///{study_dir}/{model_key}.db"

    direction = "minimize"
    sampler   = TPESampler(seed=42)
    pruner    = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    load_if_exists = "append" if resume else "create"
    if not resume:
        # Delete old study
        try:
            optuna.delete_study(study_name=model_key, storage=db_path)
        except Exception:
            pass

    study = optuna.create_study(
        study_name=model_key,
        storage=db_path,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = build_objective(model_key, model_spec, X_train, y_train, X_val, y_val)

    log.info(f"  Searching {model_key}: {n_trials} trials, {n_jobs} jobs...")
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
        catch=(Exception,),
    )

    best = study.best_trial
    log.info(f"  Best RMSE: {best.value:.4f}")
    log.info(f"  Best params: {best.params}")

    # Map unit_size back to units for DL
    best_params = dict(best.params)
    if "unit_size" in best_params:
        best_params["units"] = best_params.pop("unit_size")

    return best_params


def save_best_params(all_best: dict, path: str = "results/search/best_params.yaml"):
    """Merge with existing best params and save."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    existing.update(all_best)
    with open(path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=True)
    log.info(f"Best params saved to {path}")


def print_search_report(all_best: dict):
    log.info("\n" + "="*60)
    log.info("HYPERPARAMETER SEARCH RESULTS")
    log.info("="*60)
    for model_key, params in all_best.items():
        log.info(f"\n  [{model_key}]")
        for k, v in params.items():
            log.info(f"    {k}: {v}")


def main():
    p = argparse.ArgumentParser(description="EAS Time Series — Hyperparameter Search")
    p.add_argument("--model",    default=None, help="Single model key to search")
    p.add_argument("--models",   default=None, help="Comma-separated model keys")
    p.add_argument("--category", default=None, help="ML or DL — search all enabled in category")
    p.add_argument("--trials",   type=int, default=50, help="Optuna trials per model")
    p.add_argument("--jobs",     type=int, default=1, help="Parallel Optuna jobs (per model)")
    p.add_argument("--scenario", default="80_20", help="Scenario to optimize on (80_20/70_30/60_40)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of train to use as validation (default 0.2)")
    p.add_argument("--resume",   action="store_true", help="Resume existing Optuna study")
    p.add_argument("--data",     default="data/raw/data_train.csv")
    p.add_argument("--config",   default="config/config.yaml")
    p.add_argument("--models-config", default="config/models_config.yaml")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)

    target_col = cfg["project"]["target_column"]
    date_col   = cfg["project"]["date_column"]
    train_ratio = SCENARIO_MAP.get(args.scenario, 0.80)

    Path("results/search").mkdir(parents=True, exist_ok=True)

    # ── Load & prepare data ───────────────────────────────────────────
    log.info(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    split = int(n * train_ratio)
    train_full = df.iloc[:split].copy()
    # Validation split from within training data (last val_ratio of train)
    val_split = int(len(train_full) * (1 - args.val_ratio))
    train_df = train_full.iloc[:val_split].copy()
    val_df   = train_full.iloc[val_split:].copy()

    feat_eng = TimeSeriesFeatureEngineer(target_col=target_col, date_col=date_col)
    train_fe = feat_eng.fit_transform(train_df).dropna()
    val_fe   = feat_eng.transform(val_df, train_full).dropna()

    feat_cols = feat_eng.get_feature_columns(train_fe)
    X_train, y_train = train_fe[feat_cols], train_fe[target_col]
    X_val,   y_val   = val_fe[feat_cols],   val_fe[target_col]

    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Features: {len(feat_cols)}")

    # ── Determine which models to search ─────────────────────────────
    models_to_search = []

    if args.model:
        keys = [args.model.strip()]
    elif args.models:
        keys = [m.strip() for m in args.models.split(",")]
    else:
        keys = None  # all

    for cat in ["ml_models", "dl_models"]:
        for key, spec in mcfg["models"].get(cat, {}).items():
            if not spec.get("enabled", True):
                continue
            if not spec.get("search_space"):
                continue
            if keys and key not in keys:
                continue
            if args.category and spec.get("category") != args.category:
                continue
            models_to_search.append((key, spec))

    if not models_to_search:
        log.error("No models found to search. Check --model / --category flags.")
        sys.exit(1)

    log.info(f"Models to search: {[k for k,_ in models_to_search]}")

    # ── Run search ────────────────────────────────────────────────────
    all_best = {}
    for model_key, model_spec in models_to_search:
        log.info(f"\n{'─'*50}")
        log.info(f"Searching: {model_key}")
        best_params = run_search(
            model_key=model_key,
            model_spec=model_spec,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=args.trials,
            n_jobs=args.jobs,
            resume=args.resume,
        )
        all_best[model_key] = best_params

    # ── Save results ──────────────────────────────────────────────────
    save_best_params(all_best)
    print_search_report(all_best)
    log.info("\nDone. Run train.py with --use-best-params to use these results.")


if __name__ == "__main__":
    main()