"""
benchmarking/benchmark_runner.py
─────────────────────────────────
The heart of the benchmarking system.

Run ALL enabled models across ALL 3 scenarios in one call:
    runner = BenchmarkRunner(config_path="config/config.yaml")
    results = runner.run(df, feature_cols=..., target_col=...)
    runner.save_results()

Designed for:
  - Scalable: add new models in models_config.yaml without touching code
  - Reproducible: seeds + MLflow logging
  - Fast iteration: skip already-run models
"""
import os
import gc
import logging
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.evaluation.metrics import ResultsTable, evaluate_all
from src.visualization.plots import (
    plot_actual_vs_predicted, plot_forecast,
    plot_metrics_comparison, plot_heatmap, plot_results_table
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates the full benchmarking pipeline:

    1. Load config → identify scenarios (80/20, 70/30, 60/40)
    2. For each scenario:
       a. Split data (temporal, no shuffle)
       b. Feature engineering (leakage-safe)
       c. Fit each model
       d. Evaluate (RMSE, MAPE, MAE, R²)
       e. Plot actual vs predicted
    3. Aggregate results → heatmap, comparison bar charts
    4. Save: CSV, plots, model artifacts
    5. (Optional) MLflow run tracking
    """

    SCENARIOS = [
        {"name": "80/20 Split", "train_ratio": 0.80},
        {"name": "70/30 Split", "train_ratio": 0.70},
        {"name": "60/40 Split", "train_ratio": 0.60},
    ]

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        models_config_path: str = "config/models_config.yaml",
        use_mlflow: bool = True,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(models_config_path) as f:
            self.models_cfg = yaml.safe_load(f)

        self.target_col: str = self.cfg["project"]["target_column"]
        self.date_col: str = self.cfg["project"]["date_column"]
        self.results_dir = Path(self.cfg["paths"]["results"])
        self.plots_dir = Path(self.cfg["paths"]["plots"])
        self.metrics_dir = Path(self.cfg["paths"]["metrics"])
        self.models_dir = Path(self.cfg["paths"]["models_saved"])

        for d in [self.results_dir, self.plots_dir, self.metrics_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.results_table = ResultsTable()
        self.use_mlflow = use_mlflow
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging
        logging.basicConfig(
            level=self.cfg["logging"]["level"],
            format=self.cfg["logging"]["format"],
        )

    # ── Model instantiation from config ──────────────────────────────────
    def _build_model(self, model_key: str, model_spec: dict):
        """
        Dynamically import and instantiate a model from its config entry.
        """
        class_path: str = model_spec["class"]
        params: dict = model_spec.get("params", {})
        module_path, class_name = class_path.rsplit(".", 1)

        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(params=params)

    def _build_ensemble(self, ensemble_key: str, ensemble_spec: dict, fitted_models: dict):
        """
        Build an ensemble from already-built base model instances.
        """
        class_path = ensemble_spec["class"]
        params = ensemble_spec.get("params", {})
        base_model_keys = ensemble_spec["base_models"]

        # Gather base model instances
        base_instances = []
        for key in base_model_keys:
            # Search across ml and dl registries
            if key in fitted_models:
                base_instances.append(fitted_models[key])
            else:
                # Instantiate fresh if not pre-built
                for category in ["ml_models", "dl_models"]:
                    if key in self.models_cfg["models"].get(category, {}):
                        spec = self.models_cfg["models"][category][key]
                        base_instances.append(self._build_model(key, spec))

        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        # Ensemble classes need base_models and optionally meta_learner
        meta_key = ensemble_spec.get("meta_learner")
        if meta_key and meta_key in fitted_models:
            meta = fitted_models[meta_key]
            return cls(base_models=base_instances, meta_learner=meta, params=params)
        elif meta_key:
            for category in ["ml_models", "dl_models"]:
                if meta_key in self.models_cfg["models"].get(category, {}):
                    spec = self.models_cfg["models"][category][meta_key]
                    meta = self._build_model(meta_key, spec)
                    return cls(base_models=base_instances, meta_learner=meta, params=params)

        return cls(base_models=base_instances, params=params)

    # ── MLflow setup ──────────────────────────────────────────────────────
    def _mlflow_start(self, run_name: str):
        if not self.use_mlflow:
            return None
        try:
            import mlflow
            mlflow.set_tracking_uri(self.cfg["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.cfg["mlflow"]["experiment_name"])
            return mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            self.use_mlflow = False
            return None

    def _mlflow_log(self, metrics: dict, params: dict, run_name: str):
        if not self.use_mlflow:
            return
        try:
            import mlflow
            mlflow.log_params({"model": run_name, **params})
            mlflow.log_metrics(metrics)
        except Exception:
            pass

    # ── Core run ─────────────────────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        scenarios: Optional[List[Dict]] = None,
        skip_models: Optional[List[str]] = None,
        only_models: Optional[List[str]] = None,
    ) -> ResultsTable:
        """
        Run the full benchmark across all scenarios and all enabled models.

        Args:
            df: Full dataset with target and date columns
            feature_cols: Columns to use as features. If None, auto-detected.
            scenarios: Override default scenarios.
            skip_models: List of model names to skip.
            only_models: If set, only run these model names.

        Returns:
            ResultsTable with all results
        """
        scenarios = scenarios or self.SCENARIOS
        skip_models = skip_models or []

        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        console.print(Panel.fit(
            f"[bold cyan]🚀 EAS Time Series Benchmark[/bold cyan]\n"
            f"Run ID: [yellow]{self._run_id}[/yellow] | "
            f"Scenarios: {len(scenarios)} | "
            f"Target: [green]{self.target_col}[/green]",
            border_style="cyan"
        ))

        for scenario in scenarios:
            scenario_name = scenario["name"]
            train_ratio = scenario["train_ratio"]

            console.print(f"\n[bold]━━━ Scenario: {scenario_name} ━━━[/bold]")

            # ── 1. Temporal split ─────────────────────────────────────
            n = len(df)
            split_idx = int(n * train_ratio)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            # ── 2. Feature engineering (fit on train only) ────────────
            feat_eng = TimeSeriesFeatureEngineer(
                target_col=self.target_col,
                date_col=self.date_col,
            )
            train_fe = feat_eng.fit_transform(train_df)
            test_fe = feat_eng.transform(test_df, pd.concat([train_df, test_df], ignore_index=True))

            # Drop NaN rows from lag creation
            train_fe = train_fe.dropna()
            test_fe = test_fe.dropna()

            # Feature columns
            if feature_cols:
                feat_cols = [c for c in feature_cols if c in train_fe.columns]
            else:
                feat_cols = feat_eng.get_feature_columns(train_fe)

            X_train = train_fe[feat_cols]
            y_train = train_fe[self.target_col]
            X_test = test_fe[feat_cols]
            y_test = test_fe[self.target_col]

            logger.info(
                f"[{scenario_name}] Train: {len(X_train)} | Test: {len(X_test)} | "
                f"Features: {len(feat_cols)}"
            )

            # Build model inventory for this scenario
            fitted_models_cache: Dict[str, Any] = {}

            # ── 3. Run ML models ──────────────────────────────────────
            for model_key, model_spec in self.models_cfg["models"].get("ml_models", {}).items():
                if not model_spec.get("enabled", True):
                    continue
                if model_key in skip_models:
                    continue
                if only_models and model_key not in only_models:
                    continue

                console.print(f"  [cyan]▶ ML  [/cyan] {model_key}")
                try:
                    model = self._build_model(model_key, model_spec)
                    model.fit_timed(X_train, y_train)
                    preds = model.predict_timed(X_test)
                    self.results_table.add(
                        model_name=model.name,
                        category="ML",
                        scenario=scenario_name,
                        y_true=y_test.values,
                        y_pred=preds,
                        fit_time=model.fit_time_,
                    )
                    self._mlflow_log(
                        evaluate_all(y_test.values, preds),
                        model.get_params(), f"{model_key}_{scenario_name}"
                    )
                    # Save plot
                    plot_actual_vs_predicted(
                        train_df[self.date_col], y_train.values,
                        test_df[self.date_col], y_test.values, preds,
                        model_name=model.name, scenario=scenario_name,
                        save_path=str(self.plots_dir / f"{model_key}_{scenario_name.replace('/', '_')}.png"),
                    )
                    plt_cleanup()
                    fitted_models_cache[model_key] = model
                except Exception as e:
                    logger.error(f"  [ERROR] {model_key}: {e}")

            # ── 4. Run DL models ──────────────────────────────────────
            for model_key, model_spec in self.models_cfg["models"].get("dl_models", {}).items():
                if not model_spec.get("enabled", True):
                    continue
                if model_key in skip_models:
                    continue
                if only_models and model_key not in only_models:
                    continue

                console.print(f"  [magenta]▶ DL  [/magenta] {model_key}")
                try:
                    model = self._build_model(model_key, model_spec)
                    model.fit_timed(X_train, y_train)
                    preds = model.predict_timed(X_test)
                    self.results_table.add(
                        model_name=model.name,
                        category="DL",
                        scenario=scenario_name,
                        y_true=y_test.values,
                        y_pred=preds,
                        fit_time=model.fit_time_,
                    )
                    plot_actual_vs_predicted(
                        train_df[self.date_col], y_train.values,
                        test_df[self.date_col], y_test.values, preds,
                        model_name=model.name, scenario=scenario_name,
                        save_path=str(self.plots_dir / f"{model_key}_{scenario_name.replace('/', '_')}.png"),
                    )
                    plt_cleanup()
                    fitted_models_cache[model_key] = model
                    gc.collect()
                except Exception as e:
                    logger.error(f"  [ERROR] {model_key}: {e}")

            # ── 5. Run Ensemble models ────────────────────────────────
            for ens_key, ens_spec in self.models_cfg["models"].get("ensemble_models", {}).items():
                if not ens_spec.get("enabled", True):
                    continue
                if ens_key in skip_models:
                    continue
                if only_models and ens_key not in only_models:
                    continue

                console.print(f"  [yellow]▶ ENS [/yellow] {ens_key}")
                try:
                    ensemble = self._build_ensemble(ens_key, ens_spec, fitted_models_cache)
                    ensemble.fit_timed(X_train, y_train)
                    preds = ensemble.predict_timed(X_test)
                    self.results_table.add(
                        model_name=ensemble.name,
                        category="Ensemble",
                        scenario=scenario_name,
                        y_true=y_test.values,
                        y_pred=preds,
                        fit_time=ensemble.fit_time_,
                    )
                    plot_actual_vs_predicted(
                        train_df[self.date_col], y_train.values,
                        test_df[self.date_col], y_test.values, preds,
                        model_name=ensemble.name, scenario=scenario_name,
                        save_path=str(self.plots_dir / f"{ens_key}_{scenario_name.replace('/', '_')}.png"),
                    )
                    plt_cleanup()
                except Exception as e:
                    logger.error(f"  [ERROR] {ens_key}: {e}")

        # ── 6. Summary outputs ────────────────────────────────────────
        self.results_table.print_summary()
        return self.results_table

    def save_results(self) -> None:
        """Save CSV + summary plots."""
        self.results_table.save(str(self.metrics_dir / "results_summary.csv"))
        df = self.results_table.to_dataframe()
        if df.empty:
            return
        for metric in ["RMSE", "MAPE (%)", "MAE", "R²"]:
            plot_heatmap(df, metric=metric,
                         save_path=str(self.plots_dir / f"heatmap_{metric.replace(' ', '_').replace('²','2')}.png"))
            plt_cleanup()
        plot_results_table(df, save_path=str(self.plots_dir / "results_table.png"))
        plt_cleanup()
        logger.info(f"All results saved to {self.results_dir}")

    def best_per_scenario(self) -> pd.DataFrame:
        """Return best model per scenario (lowest RMSE)."""
        df = self.results_table.to_dataframe()
        return df.loc[df.groupby("Scenario")["RMSE"].idxmin()].reset_index(drop=True)


def plt_cleanup():
    """Release matplotlib memory."""
    import matplotlib.pyplot as plt
    plt.close("all")
    gc.collect()
