"""
benchmarking/experiment_tracker.py
────────────────────────────────────
Lightweight experiment tracking layer on top of MLflow.
Simplifies logging for notebooks and scripts.
"""
import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Thin wrapper around MLflow for consistent experiment logging.

    Usage:
        tracker = ExperimentTracker(experiment_name="eas_timeseries")
        with tracker.run("XGBoost_80_20"):
            tracker.log_params({"n_estimators": 300})
            tracker.log_metrics({"RMSE": 0.042, "MAPE": 3.21})
            tracker.log_artifact("results/plots/xgb.png")
    """

    def __init__(
        self,
        experiment_name: str = "eas_timeseries",
        tracking_uri: str = "mlruns",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._active_run = None
        self._available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            return True
        except Exception as e:
            logger.warning(f"MLflow unavailable: {e}. Tracking disabled.")
            return False

    def run(self, run_name: str):
        """Context manager for a single experiment run."""
        return _RunContext(self, run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self._available or self._active_run is None:
            return
        import mlflow
        # MLflow requires string values
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if not self._available or self._active_run is None:
            return
        import mlflow
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: str) -> None:
        if not self._available or self._active_run is None:
            return
        import mlflow
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            logger.debug(f"Artifact log skipped: {e}")

    def log_model_summary(
        self,
        model_name: str,
        scenario: str,
        metrics: Dict[str, float],
        params: Dict[str, Any],
        fit_time: float,
    ) -> None:
        """Convenience: log a full model result in one call."""
        self.log_params({"model": model_name, "scenario": scenario, **params})
        self.log_metrics({**metrics, "fit_time_s": fit_time})


class _RunContext:
    def __init__(self, tracker: ExperimentTracker, run_name: str):
        self.tracker = tracker
        self.run_name = run_name

    def __enter__(self):
        if self.tracker._available:
            import mlflow
            self.tracker._active_run = mlflow.start_run(run_name=self.run_name)
        return self.tracker

    def __exit__(self, *args):
        if self.tracker._available:
            import mlflow
            mlflow.end_run()
        self.tracker._active_run = None
