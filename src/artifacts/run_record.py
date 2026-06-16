"""
src/artifacts/run_record.py
────────────────────────────
A single experiment run record.

Every call to `ArtifactStore.start_run()` creates one RunRecord that:
  - Has a unique, timestamped run ID
  - Owns a dedicated directory under `runs/<run_id>/`
  - Snapshots config at creation time
  - Accumulates: params, metrics, model, predictions, plots
  - Finalizes to a human-readable Markdown report
  - Makes all files immutable (read-only) after finalization

Directory structure for each run:
  runs/
  └── 20260616_103045_xgboost_8020/
      ├── run_metadata.json       ← who, when, how long
      ├── config_snapshot.yaml    ← frozen config at run time
      ├── models_config_snapshot.yaml
      ├── params.json             ← model hyperparameters
      ├── metrics.json            ← RMSE, MAPE, MAE, R2, ...
      ├── predictions.csv         ← date, y_true, y_pred, residual
      ├── model.joblib            ← serialized model
      ├── plots/                  ← all plots for this run
      │   ├── actual_vs_predicted.png
      │   └── feature_importance.png
      └── run_report.md           ← human-readable summary (immutable)
"""
import json
import logging
import os
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _make_immutable(path: Path) -> None:
    """Set a file or all files under a directory to read-only."""
    if path.is_file():
        try:
            path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except Exception:
            pass  # Windows sometimes restricts chmod — non-fatal
    elif path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    f.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                except Exception:
                    pass


def _safe_serialize(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to strings."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class RunRecord:
    """
    Represents one experiment run — one model, one scenario.
    Created via ArtifactStore.start_run(); finalized via run.finish().
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        model_name: str,
        scenario: str,
        config: dict,
        models_config: dict,
    ):
        self.run_dir    = run_dir
        self.run_id     = run_id
        self.model_name = model_name
        self.scenario   = scenario
        self.config     = config
        self.models_config = models_config

        self._start_time  = time.time()
        self._start_dt    = datetime.now()
        self._params:  Dict[str, Any] = {}
        self._metrics: Dict[str, float] = {}
        self._predictions: Optional[pd.DataFrame] = None
        self._model_path:  Optional[Path] = None
        self._finalized = False
        self._status = "running"

        # Create directory layout
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

        # Immediately snapshot configs (before any training happens)
        self._dump_configs()
        self._write_metadata(status="running")
        logger.info(f"[ArtifactStore] Run started: {run_id}")

    # ── Config snapshot ────────────────────────────────────────────────────
    def _dump_configs(self) -> None:
        """Freeze config files at run start — immutable record of what was used."""
        with open(self.run_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=True)
        with open(self.run_dir / "models_config_snapshot.yaml", "w") as f:
            yaml.dump(self.models_config, f, default_flow_style=False, sort_keys=True)

    # ── Metadata ───────────────────────────────────────────────────────────
    def _write_metadata(self, status: str = "running") -> None:
        elapsed = time.time() - self._start_time
        meta = {
            "run_id":     self.run_id,
            "model_name": self.model_name,
            "scenario":   self.scenario,
            "status":     status,
            "started_at": self._start_dt.isoformat(),
            "elapsed_s":  round(elapsed, 2),
            "target_col": self.config.get("project", {}).get("target_column", "?"),
            "date_col":   self.config.get("project", {}).get("date_column", "?"),
        }
        with open(self.run_dir / "run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=_safe_serialize)

    # ── Logging API ────────────────────────────────────────────────────────
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log model hyperparameters."""
        self._params.update(params)
        with open(self.run_dir / "params.json", "w") as f:
            json.dump(self._params, f, indent=2, default=_safe_serialize)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics (RMSE, MAPE, MAE, R², etc.)."""
        self._metrics.update(metrics)
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self._metrics, f, indent=2, default=_safe_serialize)
        logger.info(
            f"  [{self.run_id}] "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    def log_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.Series] = None,
    ) -> None:
        """Save actual vs predicted values to CSV."""
        n = len(y_true)
        data = {
            "y_true":   np.array(y_true).flatten(),
            "y_pred":   np.array(y_pred).flatten(),
            "residual": np.array(y_true).flatten() - np.array(y_pred).flatten(),
            "abs_error": np.abs(np.array(y_true).flatten() - np.array(y_pred).flatten()),
        }
        if dates is not None:
            data["date"] = dates.values[:n]
        self._predictions = pd.DataFrame(data)
        self._predictions.to_csv(self.run_dir / "predictions.csv", index=False)

    def log_model(self, model) -> Path:
        """
        Save the fitted model to disk as an artifact.
        Uses joblib for sklearn/xgb/lgb; keras .keras format for DL models.
        """
        import joblib
        model_path = self.run_dir / "model.joblib"
        joblib.dump(model, model_path)
        self._model_path = model_path

        # For DL models, also save Keras weights separately
        if hasattr(model, "model") and hasattr(model.model, "save"):
            keras_path = self.run_dir / "model_keras"
            try:
                model.model.save(str(keras_path))
                logger.debug(f"  Keras model saved to {keras_path}")
            except Exception as e:
                logger.debug(f"  Keras save skipped: {e}")

        logger.info(f"  [{self.run_id}] Model saved → {model_path}")
        return model_path

    def log_plot(self, fig, name: str) -> Path:
        """Save a matplotlib figure to the run's plots/ directory."""
        import matplotlib.pyplot as plt
        plot_path = self.run_dir / "plots" / f"{name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def log_file(self, source_path: str, name: Optional[str] = None) -> Path:
        """Copy an arbitrary file into the run directory."""
        src = Path(source_path)
        dst = self.run_dir / (name or src.name)
        shutil.copy2(src, dst)
        return dst

    # ── Finalization ───────────────────────────────────────────────────────
    def finish(self, status: str = "success") -> None:
        """
        Finalize the run:
          1. Write final metadata (elapsed time, status)
          2. Generate run_report.md
          3. Make all files immutable (read-only)
        """
        if self._finalized:
            return
        self._status = status
        self._write_metadata(status=status)
        self._write_report()
        _make_immutable(self.run_dir)
        self._finalized = True
        elapsed = time.time() - self._start_time
        logger.info(
            f"[ArtifactStore] Run finished: {self.run_id} "
            f"({status}) in {elapsed:.1f}s → {self.run_dir}"
        )

    def _write_report(self) -> None:
        """Generate a human-readable Markdown report for this run."""
        elapsed = time.time() - self._start_time
        lines = []

        lines.append(f"# Run Report: `{self.run_id}`\n")
        lines.append(f"**Status**: {self._status}  ")
        lines.append(f"**Started**: {self._start_dt.strftime('%Y-%m-%d %H:%M:%S')}  ")
        lines.append(f"**Elapsed**: {elapsed:.1f}s  \n")

        lines.append("## Experiment Info\n")
        lines.append(f"| Key | Value |")
        lines.append(f"|-----|-------|")
        lines.append(f"| Model | `{self.model_name}` |")
        lines.append(f"| Scenario | {self.scenario} |")
        lines.append(f"| Target | `{self.config.get('project', {}).get('target_column', '?')}` |")
        lines.append(f"| Run ID | `{self.run_id}` |")
        lines.append("")

        if self._metrics:
            lines.append("## Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in self._metrics.items():
                lines.append(f"| {k} | `{v:.6f}` |")
            lines.append("")

        if self._params:
            lines.append("## Hyperparameters\n")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for k, v in self._params.items():
                lines.append(f"| `{k}` | `{v}` |")
            lines.append("")

        if self._predictions is not None:
            p = self._predictions
            lines.append("## Prediction Statistics\n")
            lines.append(f"| Stat | y_true | y_pred | residual |")
            lines.append(f"|------|--------|--------|----------|")
            for stat_name, fn in [("mean", np.mean), ("std", np.std),
                                   ("min", np.min), ("max", np.max)]:
                lines.append(
                    f"| {stat_name} "
                    f"| {fn(p['y_true']):.2f} "
                    f"| {fn(p['y_pred']):.2f} "
                    f"| {fn(p['residual']):.2f} |"
                )
            lines.append("")

        if self._model_path:
            lines.append("## Artifacts\n")
            for f in sorted(self.run_dir.rglob("*")):
                if f.is_file() and f.name != "run_report.md":
                    rel = f.relative_to(self.run_dir)
                    size = f.stat().st_size
                    lines.append(f"- `{rel}` ({size:,} bytes)")
            lines.append("")

        lines.append("## Config Snapshot\n")
        lines.append("```yaml")
        proj = self.config.get("project", {})
        for k, v in proj.items():
            lines.append(f"{k}: {v}")
        lines.append("```")
        lines.append("")
        lines.append("*Full config: see `config_snapshot.yaml` in this run directory.*\n")

        report_path = self.run_dir / "run_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self._metrics)

    @property
    def path(self) -> Path:
        return self.run_dir

    def __repr__(self) -> str:
        return f"RunRecord(id={self.run_id!r}, model={self.model_name!r}, status={self._status!r})"