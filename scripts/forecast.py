"""
scripts/forecast.py  (v2 — Rupiah Resilience)
──────────────────────────────────────────────
Generates submission for the Kaggle competition.

KEY DIFFERENCE from generic forecaster:
  The Kaggle test set (data_test.csv) already contains all exogenous features
  (OIL, GOLD, SP500, IHSG, VIX, CPI, BI_rate, US_rate) for the forecast period.
  So we do NOT need iterative one-step-ahead forecasting.
  Instead: train on full data_train.csv, predict on data_test.csv directly.

  For lag features: we prepend train data context to test before computing features.

Usage:
  python scripts/forecast.py --model xgboost
  python scripts/forecast.py --model xgboost --use-best-params
  python scripts/forecast.py --model lstm
  python scripts/forecast.py --ensemble voting_xgb_lstm
"""
import argparse
import importlib
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.visualization.plots import plot_forecast
from src.artifacts import ArtifactStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/forecast.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("forecast")


def prepare_submission_features(
    train_df: pd.DataFrame,
    kaggle_test_df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> tuple:
    """
    Prepare feature matrices for:
      1. Full training (X_full, y_full) — for final model training
      2. Kaggle test  (X_test_kaggle)   — for submission predictions

    Strategy:
      - Feature engineer full train data
      - Append test rows (NaN target) to compute lag features from train context
      - Return test feature rows separately

    Returns:
        (X_full, y_full, X_kaggle, feat_eng, feat_cols)
    """
    feat_eng = TimeSeriesFeatureEngineer(
        target_col=target_col,
        date_col=date_col,
    )

    # 1. Feature engineer train
    train_fe = feat_eng.fit_transform(train_df).dropna()
    feat_cols = feat_eng.get_feature_columns(train_fe)
    X_full = train_fe[feat_cols]
    y_full = train_fe[target_col]
    log.info(f"Full train features: {X_full.shape}")

    # 2. Feature engineer Kaggle test (with train context for lags)
    test_with_nan = kaggle_test_df.copy()
    test_with_nan[target_col] = np.nan

    # Align columns (test may be missing target)
    train_cols = [c for c in train_df.columns if c in test_with_nan.columns or c == target_col]
    test_aligned = test_with_nan.reindex(columns=train_df.columns)

    full_combined = pd.concat([train_df, test_aligned], ignore_index=True)
    full_fe = feat_eng.fit_transform(full_combined)

    # Extract only the test portion
    test_start_idx = len(train_df)
    test_fe = full_fe.iloc[test_start_idx:].copy()
    test_fe = test_fe.dropna(subset=[c for c in feat_cols if "lag" not in c])  # keep lag NaNs filled

    # Ensure feat_cols exist in test
    available_cols = [c for c in feat_cols if c in test_fe.columns]
    X_kaggle = test_fe[available_cols].fillna(method="ffill").fillna(0)

    log.info(f"Kaggle test features: {X_kaggle.shape}")
    return X_full, y_full, X_kaggle, feat_eng, available_cols


def build_model(model_key, model_spec, params_override=None):
    params = model_spec.get("params", {}).copy()
    if params_override:
        params.update(params_override)
    module_path, class_name = model_spec["class"].rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)(params=params)


def main():
    p = argparse.ArgumentParser(description="Rupiah Resilience — Submission Generator")
    p.add_argument("--model",        default=None, help="Model key from config")
    p.add_argument("--ensemble",     default=None, help="Ensemble key from config")
    p.add_argument("--use-best-params", action="store_true")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--models-config", default="config/models_config.yaml")
    p.add_argument("--out", default="data/submissions/submission.csv",
                   help="Output submission CSV path")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)

    target_col = cfg["project"]["target_column"]   # USDIDR
    date_col   = cfg["project"]["date_column"]      # Date
    plots_dir  = Path(cfg["paths"]["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    loader = DataLoader(args.config)
    train_df      = loader.load_train()
    kaggle_test_df = loader.load_kaggle_test()

    log.info(f"Train: {train_df.shape} | Kaggle test: {kaggle_test_df.shape}")

    # ── Prepare features ──────────────────────────────────────────────
    X_full, y_full, X_kaggle, feat_eng, feat_cols = prepare_submission_features(
        train_df, kaggle_test_df, target_col, date_col
    )

    # ── Load best params if requested ─────────────────────────────────
    best_params_all = {}
    if args.use_best_params:
        bp_path = "results/search/best_params.yaml"
        if Path(bp_path).exists():
            with open(bp_path, encoding="utf-8") as f:
                best_params_all = yaml.safe_load(f) or {}
            log.info(f"Loaded best params for: {list(best_params_all.keys())}")
        else:
            log.warning("No best_params.yaml found — using default params")

    # ── Build & train model ───────────────────────────────────────────
    if args.model:
        model_key = args.model
        spec = None
        for cat in ["ml_models", "dl_models"]:
            spec = mcfg["models"].get(cat, {}).get(model_key)
            if spec:
                break
        if not spec:
            log.error(f"Model '{model_key}' not found in config.")
            sys.exit(1)

        model = build_model(model_key, spec, best_params_all.get(model_key))
        log.info(f"Training {model_key} on full dataset ({len(X_full)} rows)...")
        model.fit_timed(X_full, y_full)
        log.info(f"Trained in {model.fit_time_:.2f}s")
        preds = model.predict(X_kaggle)
        model_label = model.name

    elif args.ensemble:
        ens_key  = args.ensemble
        ens_spec = mcfg["models"]["ensemble_models"].get(ens_key)
        if not ens_spec:
            log.error(f"Ensemble '{ens_key}' not found.")
            sys.exit(1)

        base_instances = []
        for k in ens_spec["base_models"]:
            for cat in ["ml_models", "dl_models"]:
                bspec = mcfg["models"].get(cat, {}).get(k)
                if bspec:
                    base_instances.append(build_model(k, bspec, best_params_all.get(k)))

        meta_key = ens_spec.get("meta_learner")
        meta_instance = None
        if meta_key:
            for cat in ["ml_models", "dl_models"]:
                ms = mcfg["models"].get(cat, {}).get(meta_key)
                if ms:
                    meta_instance = build_model(meta_key, ms)

        module_path, class_name = ens_spec["class"].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        weights = ens_spec.get("params", {}).get("weights")

        if meta_instance:
            model = cls(base_models=base_instances, meta_learner=meta_instance)
        elif weights:
            model = cls(base_models=base_instances, weights=weights)
        else:
            model = cls(base_models=base_instances)

        log.info(f"Training ensemble {ens_key} on full dataset...")
        model.fit_timed(X_full, y_full)
        preds = model.predict(X_kaggle)
        model_label = ens_key
    else:
        log.error("Provide --model or --ensemble.")
        sys.exit(1)

    # ── Validate predictions ──────────────────────────────────────────
    preds = np.array(preds).flatten()
    log.info(f"Predictions: {len(preds)} rows")
    log.info(f"  Range : {preds.min():.0f} – {preds.max():.0f}")
    log.info(f"  Mean  : {preds.mean():.0f}")
    log.info(f"  Std   : {preds.std():.0f}")

    # Sanity check: USD/IDR should be in range ~10,000 – 20,000
    if preds.min() < 5000 or preds.max() > 30000:
        log.warning(f"WARNING: Predictions outside expected USD/IDR range [5000, 30000]")
        log.warning("Consider checking model training or feature engineering.")

    # ── Build submission CSV ──────────────────────────────────────────
    dates = kaggle_test_df[date_col]
    sub_df = loader.build_submission(
        predictions=pd.Series(preds),
        dates=dates,
        output_path=args.out,
    )

    # ── Record as artifact ────────────────────────────────────────────
    store = ArtifactStore(args.config, args.models_config)
    run = store.start_run(
        model_name=model_label,
        scenario="Kaggle Submission (full train)",
    )
    run.log_params(model.get_params() if hasattr(model, "get_params") else {})
    run.log_metrics({
        "n_predictions": float(len(preds)),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
    })
    run.log_predictions(
        y_true=np.zeros(len(preds)),   # unknown at submission time
        y_pred=preds,
        dates=dates,
    )
    run.log_model(model)
    run.log_file(args.out, name="submission.csv")

    # ── Plot: Historical + Forecast ───────────────────────────────────
    plot_path = str(plots_dir / f"forecast_{model_label}.png")
    try:
        fig = plot_forecast(
            train_dates=train_df[date_col],
            train_actual=train_df[target_col].values,
            forecast_dates=sub_df[date_col],
            forecast_values=sub_df[target_col].values,
            model_name=model_label,
            save_path=plot_path,
        )
        plt.close("all")
        log.info(f"Forecast plot: {plot_path}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    log.info(f"\nSubmission: {args.out}")
    log.info(f"  Rows    : {len(sub_df)}")
    log.info(f"  Period  : {sub_df[date_col].iloc[0].date()} -> {sub_df[date_col].iloc[-1].date()}")
    log.info(f"  USDIDR  : {preds.min():.0f} - {preds.max():.0f} (mean={preds.mean():.0f})")

    # Save plot into run artifact too
    try:
        plot_file = plots_dir / f"forecast_{model_label}.png"
        if plot_file.exists():
            run.log_file(str(plot_file), name="forecast_plot.png")
    except Exception:
        pass

    run.finish(status="success")
    store.rebuild_index()
    log.info(f"Artifact run: {run.path}")


if __name__ == "__main__":
    main()
