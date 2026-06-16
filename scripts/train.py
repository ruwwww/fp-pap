"""
scripts/train.py  (v3 — with ArtifactStore)
─────────────────────────────────────────────
Pure Python training script — runs ALL models across all 3 scenarios.
Every model × scenario combination produces an immutable artifact in runs/.

Artifact per run:
  runs/<timestamp>_<model>_<scenario>/
    ├── config_snapshot.yaml        ← frozen config at this exact run
    ├── models_config_snapshot.yaml
    ├── run_metadata.json           ← who/when/status/elapsed
    ├── params.json                 ← hyperparameters used
    ├── metrics.json                ← RMSE, MAPE, MAE, R2
    ├── predictions.csv             ← y_true, y_pred, residual, date
    ├── model.joblib                ← serialized model
    ├── plots/
    │   └── actual_vs_predicted.png
    └── run_report.md               ← human-readable summary (immutable)

Usage:
  python scripts/train.py
  python scripts/train.py --parallel --workers 4
  python scripts/train.py --models xgboost,lightgbm
  python scripts/train.py --skip-category DL --scenarios 80_20
  python scripts/train.py --use-best-params
"""
import argparse
import gc
import importlib
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.evaluation.metrics import ResultsTable, evaluate_all
from src.artifacts import ArtifactStore
from src.visualization.plots import plot_actual_vs_predicted, plot_heatmap, plot_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/train.log", mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("train")

SCENARIOS = [
    {"key": "80_20", "label": "80/20 Split", "train_ratio": 0.80},
    {"key": "70_30", "label": "70/30 Split", "train_ratio": 0.70},
    {"key": "60_40", "label": "60/40 Split", "train_ratio": 0.60},
]


# ── Subprocess worker (must be top-level for pickle) ────────────────────────
def _parallel_worker(
    model_key: str,
    model_spec: dict,
    X_tr_path: str,
    y_tr_path: str,
    X_te_path: str,
    y_te_path: str,
    scenario_label: str,
    best_params_path: Optional[str],
) -> dict:
    """Train a single model in a subprocess, return results dict."""
    import warnings, sys, os, time, gc, importlib
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import numpy as np
    import pandas as pd
    import yaml
    from src.evaluation.metrics import evaluate_all

    try:
        X_train = pd.read_parquet(X_tr_path)
        y_train = pd.read_parquet(y_tr_path).squeeze()
        X_test  = pd.read_parquet(X_te_path)
        y_test  = pd.read_parquet(y_te_path).squeeze()

        params = model_spec.get("params", {}).copy()
        if best_params_path and Path(best_params_path).exists():
            with open(best_params_path, encoding="utf-8") as f:
                best = yaml.safe_load(f) or {}
            if model_key in best:
                params.update(best[model_key])

        module_path, class_name = model_spec["class"].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        model = getattr(mod, class_name)(params=params)

        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        preds = model.predict(X_test)
        metrics = evaluate_all(y_test.values, preds)
        gc.collect()

        return {
            "status":     "ok",
            "model_key":  model_key,
            "model_name": model.name,
            "category":   model_spec.get("category", "ML"),
            "scenario":   scenario_label,
            "params":     params,
            "metrics":    metrics,
            "fit_time":   fit_time,
            "preds":      preds.tolist(),
            "y_test":     y_test.values.tolist(),
        }
    except Exception as e:
        import traceback
        return {
            "status":    "error",
            "model_key": model_key,
            "scenario":  scenario_label,
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }


class TrainingPipeline:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        models_config_path: str = "config/models_config.yaml",
    ):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        with open(models_config_path, encoding="utf-8") as f:
            self.mcfg = yaml.safe_load(f)

        self.config_path = config_path
        self.models_config_path = models_config_path
        self.target_col  = self.cfg["project"]["target_column"]
        self.date_col    = self.cfg["project"]["date_column"]
        self.plots_dir   = Path(self.cfg["paths"]["plots"])
        self.metrics_dir = Path(self.cfg["paths"]["metrics"])
        self.tmp_dir     = Path("results/.tmp")

        for d in [self.plots_dir, self.metrics_dir, self.tmp_dir,
                  Path("results"), Path("runs")]:
            d.mkdir(parents=True, exist_ok=True)

        self.results  = ResultsTable()
        self.store    = ArtifactStore(
            config_path=config_path,
            models_config_path=models_config_path,
        )

    # ── Helpers ──────────────────────────────────────────────────────────
    def _load_data(self, data_path: str) -> pd.DataFrame:
        loader = DataLoader(self.config_path)
        df = loader.load_csv(data_path)
        log.info(f"Loaded: {df.shape} | cols: {list(df.columns)}")
        return df

    def _prepare_split(self, df: pd.DataFrame, train_ratio: float):
        n = len(df)
        split = int(n * train_ratio)
        train_df = df.iloc[:split].copy().reset_index(drop=True)
        test_df  = df.iloc[split:].copy().reset_index(drop=True)

        feat_eng = TimeSeriesFeatureEngineer(
            target_col=self.target_col, date_col=self.date_col
        )
        train_fe = feat_eng.fit_transform(train_df).dropna()
        full_ctx = pd.concat([train_df, test_df], ignore_index=True)
        test_fe  = feat_eng.transform(test_df, full_ctx).dropna()

        feat_cols = feat_eng.get_feature_columns(train_fe)

        X_train = train_fe[feat_cols]
        y_train = train_fe[self.target_col]
        X_test  = test_fe[feat_cols]
        y_test  = test_fe[self.target_col]

        # Persist for parallel workers
        suffix = f"{int(train_ratio*100)}"
        X_train.to_parquet(self.tmp_dir / f"X_train_{suffix}.parquet")
        y_train.to_frame().to_parquet(self.tmp_dir / f"y_train_{suffix}.parquet")
        X_test.to_parquet(self.tmp_dir / f"X_test_{suffix}.parquet")
        y_test.to_frame().to_parquet(self.tmp_dir / f"y_test_{suffix}.parquet")

        return (
            str(self.tmp_dir / f"X_train_{suffix}.parquet"),
            str(self.tmp_dir / f"y_train_{suffix}.parquet"),
            str(self.tmp_dir / f"X_test_{suffix}.parquet"),
            str(self.tmp_dir / f"y_test_{suffix}.parquet"),
            train_df, test_df, feat_eng, feat_cols,
            X_train, y_train, X_test, y_test,
        )

    def _load_best_params(self) -> Optional[str]:
        path = "results/search/best_params.yaml"
        return path if Path(path).exists() else None

    def _build_model(self, model_key, model_spec, best_params_path):
        params = model_spec.get("params", {}).copy()
        if best_params_path and Path(best_params_path).exists():
            with open(best_params_path, encoding="utf-8") as f:
                best = yaml.safe_load(f) or {}
            if model_key in best:
                params.update(best[model_key])
                log.info(f"    Using searched params for {model_key}")
        module_path, class_name = model_spec["class"].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)(params=params), params

    def _get_enabled_models(self, filter_keys, skip_keys, skip_categories):
        models = []
        for cat in ["ml_models", "dl_models"]:
            for key, spec in self.mcfg["models"].get(cat, {}).items():
                if not spec.get("enabled", True): continue
                if filter_keys and key not in filter_keys: continue
                if skip_keys and key in skip_keys: continue
                if skip_categories and spec.get("category") in skip_categories: continue
                models.append((key, spec))
        return models

    # ── Single model train + artifact ─────────────────────────────────────
    def _train_and_record(
        self,
        model_key: str,
        model_spec: dict,
        model_name: str,
        category: str,
        params: dict,
        metrics: dict,
        fit_time: float,
        preds: np.ndarray,
        scenario_label: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_obj=None,
    ) -> None:
        """Record results into ArtifactStore and ResultsTable."""
        # ── ArtifactStore ─────────────────────────────────────────────
        run = self.store.start_run(
            model_name=model_name,
            scenario=scenario_label,
        )
        run.log_params({"fit_time_s": fit_time, **params})
        run.log_metrics(metrics)
        run.log_predictions(
            y_true=y_test.values,
            y_pred=preds,
            dates=test_df[self.date_col] if self.date_col in test_df.columns else None,
        )
        if model_obj is not None:
            run.log_model(model_obj)

        # ── Actual vs Predicted plot ──────────────────────────────────
        try:
            fig = plot_actual_vs_predicted(
                train_df[self.date_col], y_train.values,
                test_df[self.date_col], y_test.values, preds,
                model_name=model_name, scenario=scenario_label,
            )
            run.log_plot(fig, "actual_vs_predicted")
            # Also save to global plots dir
            safe = scenario_label.replace("/", "_").replace(" ", "_")
            fig.savefig(str(self.plots_dir / f"{model_key}_{safe}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            log.debug(f"Plot failed: {e}")

        run.finish(status="success")

        # ── ResultsTable ──────────────────────────────────────────────
        self.results.add(
            model_name=model_name,
            category=category,
            scenario=scenario_label,
            y_true=y_test.values,
            y_pred=preds,
            fit_time=fit_time,
        )

        log.info(
            f"  [{scenario_label}] {model_name:22s} | "
            f"RMSE={metrics['RMSE']:.2f} | MAPE={metrics['MAPE']:.3f}% | "
            f"R2={metrics['R2']:.4f} | {fit_time:.1f}s | run={run.run_id[:30]}"
        )

    # ── Main run ──────────────────────────────────────────────────────────
    def run(
        self,
        data_path: str,
        parallel: bool = False,
        workers: int = 4,
        use_best_params: bool = False,
        filter_models: Optional[List[str]] = None,
        skip_models: Optional[List[str]] = None,
        skip_categories: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
    ) -> ResultsTable:
        df = self._load_data(data_path)
        scenarios_to_run = [s for s in SCENARIOS
                            if scenarios is None or s["key"] in scenarios]
        best_params_path = self._load_best_params() if use_best_params else None
        all_models = self._get_enabled_models(filter_models, skip_models, skip_categories)

        log.info(f"Models: {[k for k,_ in all_models]}")
        log.info(f"Scenarios: {[s['label'] for s in scenarios_to_run]}")
        log.info(f"Parallel: {parallel} | Use best params: {use_best_params}")

        for scenario in scenarios_to_run:
            label = scenario["label"]
            ratio = scenario["train_ratio"]
            log.info(f"\n{'='*65}\nScenario: {label}\n{'='*65}")

            (X_tr_path, y_tr_path, X_te_path, y_te_path,
             train_df, test_df, feat_eng, feat_cols,
             X_train, y_train, X_test, y_test) = self._prepare_split(df, ratio)

            log.info(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feat_cols)}")

            parallel_models  = [(k, s) for k, s in all_models if s.get("parallel_safe", True)]
            sequential_models = [(k, s) for k, s in all_models if not s.get("parallel_safe", True)]
            ensemble_specs   = {
                k: s for k, s in self.mcfg["models"].get("ensemble_models", {}).items()
                if s.get("enabled", True) and (filter_models is None or k in filter_models)
            }

            # ── PARALLEL ML ───────────────────────────────────────────
            if parallel and parallel_models:
                log.info(f"  Running {len(parallel_models)} ML models in parallel ({workers} workers)...")
                futures = {}
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for model_key, model_spec in parallel_models:
                        future = executor.submit(
                            _parallel_worker,
                            model_key, model_spec,
                            X_tr_path, y_tr_path, X_te_path, y_te_path,
                            label, best_params_path,
                        )
                        futures[future] = (model_key, model_spec)

                    for future in as_completed(futures):
                        model_key, model_spec = futures[future]
                        result = future.result()
                        if result["status"] == "ok":
                            # Can't save model object from subprocess, mark path as None
                            self._train_and_record(
                                model_key=model_key,
                                model_spec=model_spec,
                                model_name=result["model_name"],
                                category=result["category"],
                                params=result["params"],
                                metrics=result["metrics"],
                                fit_time=result["fit_time"],
                                preds=np.array(result["preds"]),
                                scenario_label=label,
                                train_df=train_df, test_df=test_df,
                                y_train=y_train, y_test=y_test,
                                model_obj=None,  # subprocess — no model obj
                            )
                        else:
                            log.error(f"  [FAIL] {model_key}: {result['error']}")
            else:
                # Sequential ML
                for model_key, model_spec in parallel_models:
                    log.info(f"  [ML] {model_key}")
                    try:
                        model, params = self._build_model(model_key, model_spec, best_params_path)
                        t0 = time.time()
                        model.fit(X_train, y_train)
                        fit_time = time.time() - t0
                        preds = model.predict(X_test)
                        metrics = evaluate_all(y_test.values, preds)
                        self._train_and_record(
                            model_key, model_spec, model.name,
                            model_spec.get("category", "ML"),
                            params, metrics, fit_time, preds, label,
                            train_df, test_df, y_train, y_test,
                            model_obj=model,
                        )
                    except Exception as e:
                        log.error(f"  [FAIL] {model_key}: {e}")
                    gc.collect()

            # ── SEQUENTIAL DL (TF global state) ──────────────────────
            for model_key, model_spec in sequential_models:
                log.info(f"  [DL] {model_key}")
                try:
                    model, params = self._build_model(model_key, model_spec, best_params_path)
                    t0 = time.time()
                    model.fit(X_train, y_train)
                    fit_time = time.time() - t0
                    preds = model.predict(X_test)
                    metrics = evaluate_all(y_test.values, preds)
                    self._train_and_record(
                        model_key, model_spec, model.name,
                        model_spec.get("category", "DL"),
                        params, metrics, fit_time, preds, label,
                        train_df, test_df, y_train, y_test,
                        model_obj=model,
                    )
                except Exception as e:
                    log.error(f"  [FAIL] {model_key}: {e}")
                gc.collect()

            # ── ENSEMBLE ──────────────────────────────────────────────
            for ens_key, ens_spec in ensemble_specs.items():
                log.info(f"  [Ensemble] {ens_key}")
                try:
                    base_instances = []
                    for k in ens_spec["base_models"]:
                        for cat in ["ml_models", "dl_models"]:
                            bspec = self.mcfg["models"].get(cat, {}).get(k)
                            if bspec:
                                m, _ = self._build_model(k, bspec, best_params_path)
                                base_instances.append(m)

                    meta_key = ens_spec.get("meta_learner")
                    meta_instance = None
                    if meta_key:
                        for cat in ["ml_models", "dl_models"]:
                            ms = self.mcfg["models"].get(cat, {}).get(meta_key)
                            if ms:
                                meta_instance, _ = self._build_model(meta_key, ms, None)

                    module_path, class_name = ens_spec["class"].rsplit(".", 1)
                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, class_name)
                    weights = ens_spec.get("params", {}).get("weights")

                    if meta_instance:
                        ensemble = cls(base_models=base_instances,
                                       meta_learner=meta_instance,
                                       params=ens_spec.get("params", {}))
                    elif weights:
                        ensemble = cls(base_models=base_instances,
                                       weights=weights,
                                       params=ens_spec.get("params", {}))
                    else:
                        ensemble = cls(base_models=base_instances,
                                       params=ens_spec.get("params", {}))

                    t0 = time.time()
                    ensemble.fit(X_train, y_train)
                    fit_time = time.time() - t0
                    preds = ensemble.predict(X_test)
                    metrics = evaluate_all(y_test.values, preds)
                    self._train_and_record(
                        ens_key, ens_spec, ensemble.name,
                        "Ensemble",
                        ens_spec.get("params", {}),
                        metrics, fit_time, preds, label,
                        train_df, test_df, y_train, y_test,
                        model_obj=ensemble,
                    )
                except Exception as e:
                    log.error(f"  [FAIL] ensemble {ens_key}: {e}")
                gc.collect()

        # ── Finalize ──────────────────────────────────────────────────
        self._finalize()
        return self.results

    def _finalize(self):
        """Save global results + rebuild artifact index."""
        self.results.save(str(self.metrics_dir / "results_summary.csv"))
        df = self.results.to_dataframe()
        if df.empty:
            return
        for metric_col in ["RMSE", "MAPE (%)", "MAE", "R²"]:
            if metric_col in df.columns:
                safe = metric_col.replace(" ", "_").replace("²","2").replace("(","").replace(")","")
                try:
                    fig = plot_heatmap(df, metric=metric_col,
                                       save_path=str(self.plots_dir / f"heatmap_{safe}.png"))
                    plt.close(fig)
                except Exception:
                    pass
        try:
            fig = plot_results_table(df, save_path=str(self.plots_dir / "results_table.png"))
            plt.close(fig)
        except Exception:
            pass

        # Rebuild artifact index
        index_path = self.store.rebuild_index()
        log.info(f"Artifact index → {index_path}")

        log.info(f"\nResults → {self.metrics_dir / 'results_summary.csv'}")
        best = df.loc[df.groupby("Scenario")["RMSE"].idxmin()]
        log.info("\n--- Best per Scenario (RMSE) ---")
        log.info(best[["Scenario", "Model", "RMSE", "MAPE (%)", "MAE", "R²"]].to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="EAS Time Series — Training Pipeline")
    p.add_argument("--data",          default="data/raw/data_train.csv")
    p.add_argument("--models",        default=None, help="Comma-separated model keys")
    p.add_argument("--skip",          default=None, help="Comma-separated model keys to skip")
    p.add_argument("--skip-category", default=None, help="ML or DL")
    p.add_argument("--scenarios",     default=None, help="e.g. 80_20,70_30")
    p.add_argument("--parallel",      action="store_true")
    p.add_argument("--workers",       type=int, default=4)
    p.add_argument("--use-best-params", action="store_true")
    p.add_argument("--config",        default="config/config.yaml")
    p.add_argument("--models-config", default="config/models_config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = TrainingPipeline(args.config, args.models_config)
    pipeline.run(
        data_path=args.data,
        parallel=args.parallel,
        workers=args.workers,
        use_best_params=args.use_best_params,
        filter_models=[m.strip() for m in args.models.split(",")] if args.models else None,
        skip_models=[m.strip() for m in args.skip.split(",")] if args.skip else None,
        skip_categories=[args.skip_category] if args.skip_category else None,
        scenarios=[s.strip() for s in args.scenarios.split(",")] if args.scenarios else None,
    )


if __name__ == "__main__":
    main()