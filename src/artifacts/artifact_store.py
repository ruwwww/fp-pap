"""
src/artifacts/artifact_store.py
─────────────────────────────────
Central registry for all experiment runs.

Usage:
    store = ArtifactStore()

    run = store.start_run(
        model_name="XGBoost",
        scenario="80/20 Split",
    )
    run.log_params(model.get_params())
    run.log_model(model)
    run.log_metrics(evaluate_all(y_test, preds))
    run.log_predictions(y_test, preds, dates=test_df["Date"])
    run.log_plot(fig, name="actual_vs_predicted")
    run.finish()

    store.print_index()       # print all runs
    store.rebuild_index()     # regenerate runs/index.md
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from .run_record import RunRecord

logger = logging.getLogger(__name__)

_RUNS_ROOT = Path("runs")


def _make_run_id(model_name: str, scenario: str) -> str:
    """
    Generate a unique, sortable run ID.
    Format: YYYYMMDD_HHMMSS_<model>_<scenario>
    Example: 20260616_103045_xgboost_8020
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize names: lowercase, alphanumeric + underscore only
    safe_model    = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    safe_scenario = re.sub(r"[^a-z0-9]+", "_", scenario.lower()).strip("_")
    return f"{ts}_{safe_model}_{safe_scenario}"


class ArtifactStore:
    """
    Manages all experiment run artifacts.

    Responsibilities:
    - Generate unique run IDs
    - Create isolated run directories
    - Maintain a global index of all runs
    - Load past runs for comparison
    """

    def __init__(
        self,
        runs_root: str = "runs",
        config_path: str = "config/config.yaml",
        models_config_path: str = "config/models_config.yaml",
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.config_path        = config_path
        self.models_config_path = models_config_path

        self._config        = self._load_yaml(config_path)
        self._models_config = self._load_yaml(models_config_path)

        self._active_runs: List[RunRecord] = []

    @staticmethod
    def _load_yaml(path: str) -> dict:
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ── Run lifecycle ──────────────────────────────────────────────────────
    def start_run(
        self,
        model_name: str,
        scenario: str,
        extra_config: Optional[dict] = None,
    ) -> RunRecord:
        """
        Create and register a new run.

        Args:
            model_name: Human-readable model name (e.g. "XGBoost")
            scenario:   Scenario label (e.g. "80/20 Split")
            extra_config: Additional key-value pairs to merge into config snapshot

        Returns:
            RunRecord — use this to log everything for this run
        """
        # Re-read configs fresh at run start (captures any edits since startup)
        config        = self._load_yaml(self.config_path)
        models_config = self._load_yaml(self.models_config_path)
        if extra_config:
            config = {**config, **extra_config}

        run_id  = _make_run_id(model_name, scenario)
        run_dir = self.runs_root / run_id

        run = RunRecord(
            run_dir=run_dir,
            run_id=run_id,
            model_name=model_name,
            scenario=scenario,
            config=config,
            models_config=models_config,
        )
        self._active_runs.append(run)
        return run

    # ── Index management ───────────────────────────────────────────────────
    def rebuild_index(self) -> Path:
        """
        Scan all run directories and regenerate `runs/index.md`.
        Sorted by run ID (newest last) with a summary table.
        """
        runs = self._scan_runs()
        index_path = self.runs_root / "index.md"

        # Make index writable if it existed (index is the one mutable file)
        if index_path.exists():
            try:
                import stat
                index_path.chmod(stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass

        lines = [
            "# Experiment Runs Index\n",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  ",
            f"*Total runs: {len(runs)}*\n",
            "| Run ID | Model | Scenario | Status | RMSE | MAPE(%) | MAE | R² | Elapsed(s) | Date |",
            "|--------|-------|----------|--------|------|---------|-----|-----|-----------|------|",
        ]

        for r in sorted(runs, key=lambda x: x["run_id"]):
            m = r.get("metrics", {})
            meta = r.get("meta", {})
            lines.append(
                f"| `{r['run_id']}` "
                f"| {r['model_name']} "
                f"| {r['scenario']} "
                f"| {meta.get('status', '?')} "
                f"| {m.get('RMSE', 'N/A'):.4f} " if isinstance(m.get("RMSE"), float) else
                f"| `{r['run_id']}` | {r['model_name']} | {r['scenario']} | {meta.get('status','?')} | - | - | - | - | - | - |"
            )
            if isinstance(m.get("RMSE"), float):
                lines[-1] = (
                    f"| [{r['run_id']}](./{r['run_id']}/run_report.md) "
                    f"| {r['model_name']} "
                    f"| {r['scenario']} "
                    f"| {meta.get('status', '?')} "
                    f"| {m.get('RMSE', 0):.4f} "
                    f"| {m.get('MAPE', 0):.3f} "
                    f"| {m.get('MAE', 0):.4f} "
                    f"| {m.get('R2', 0):.4f} "
                    f"| {meta.get('elapsed_s', '?')} "
                    f"| {meta.get('started_at', '?')[:10]} |"
                )

        lines.append("")
        lines.append("---")
        lines.append("\n## Quick Stats\n")

        # Best per scenario
        df_runs = pd.DataFrame([
            {
                "run_id": r["run_id"],
                "model": r["model_name"],
                "scenario": r["scenario"],
                **r.get("metrics", {}),
            }
            for r in runs
            if r.get("metrics", {}).get("RMSE") is not None
        ])
        if not df_runs.empty and "RMSE" in df_runs.columns:
            lines.append("### Best RMSE per Scenario\n")
            lines.append("| Scenario | Best Model | RMSE | MAPE(%) | R² |")
            lines.append("|----------|-----------|------|---------|-----|")
            for scenario, grp in df_runs.groupby("scenario"):
                best = grp.sort_values("RMSE").iloc[0]
                lines.append(
                    f"| {scenario} | {best['model']} "
                    f"| {best['RMSE']:.4f} "
                    f"| {best.get('MAPE', 0):.3f} "
                    f"| {best.get('R2', 0):.4f} |"
                )

        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"[ArtifactStore] Index updated → {index_path} ({len(runs)} runs)")
        return index_path

    def _scan_runs(self) -> List[dict]:
        """Scan all run directories and load their metadata + metrics."""
        results = []
        for run_dir in sorted(self.runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_path    = run_dir / "run_metadata.json"
            metrics_path = run_dir / "metrics.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                metrics = {}
                if metrics_path.exists():
                    with open(metrics_path, encoding="utf-8") as f:
                        metrics = json.load(f)
                results.append({
                    "run_id":     run_dir.name,
                    "model_name": meta.get("model_name", "?"),
                    "scenario":   meta.get("scenario", "?"),
                    "meta":       meta,
                    "metrics":    metrics,
                    "path":       str(run_dir),
                })
            except Exception as e:
                logger.debug(f"Could not read run {run_dir.name}: {e}")
        return results

    def print_index(self, top_n: int = 20) -> None:
        """Print a summary of recent runs to console."""
        runs = self._scan_runs()
        if not runs:
            print("No runs found.")
            return

        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title=f"Experiment Runs ({len(runs)} total)", show_lines=True)
            table.add_column("Run ID", style="cyan", no_wrap=True)
            table.add_column("Model", style="green")
            table.add_column("Scenario")
            table.add_column("Status")
            table.add_column("RMSE", justify="right")
            table.add_column("MAPE%", justify="right")
            table.add_column("R²", justify="right")
            table.add_column("Elapsed(s)", justify="right")

            for r in sorted(runs, reverse=True)[:top_n]:
                m = r.get("metrics", {})
                meta = r.get("meta", {})
                table.add_row(
                    r["run_id"][:30],
                    r["model_name"],
                    r["scenario"],
                    meta.get("status", "?"),
                    f"{m['RMSE']:.4f}" if "RMSE" in m else "-",
                    f"{m['MAPE']:.3f}" if "MAPE" in m else "-",
                    f"{m['R2']:.4f}"   if "R2"   in m else "-",
                    str(meta.get("elapsed_s", "?")),
                )
            console.print(table)
        except ImportError:
            for r in runs[-top_n:]:
                m = r.get("metrics", {})
                print(f"{r['run_id']:45s} | {r['model_name']:20s} | "
                      f"RMSE={m.get('RMSE','?')}")

    def load_run(self, run_id: str) -> Optional[dict]:
        """Load all artifacts from a past run by ID."""
        run_dir = self.runs_root / run_id
        if not run_dir.exists():
            logger.error(f"Run not found: {run_id}")
            return None

        result = {"run_id": run_id, "path": run_dir}

        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                result["meta"] = json.load(f)

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                result["metrics"] = json.load(f)

        params_path = run_dir / "params.json"
        if params_path.exists():
            with open(params_path, encoding="utf-8") as f:
                result["params"] = json.load(f)

        preds_path = run_dir / "predictions.csv"
        if preds_path.exists():
            result["predictions"] = pd.read_csv(preds_path)

        config_path = run_dir / "config_snapshot.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                result["config"] = yaml.safe_load(f)

        model_path = run_dir / "model.joblib"
        if model_path.exists():
            result["model_path"] = str(model_path)

        result["plots"] = list((run_dir / "plots").glob("*.png")) if (run_dir / "plots").exists() else []
        return result

    def load_model(self, run_id: str):
        """Load the saved model from a past run."""
        import joblib
        model_path = self.runs_root / run_id / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No model.joblib in run {run_id}")
        return joblib.load(model_path)

    def compare_runs(self, run_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Return a DataFrame comparing metrics across runs (or all runs)."""
        runs = self._scan_runs()
        if run_ids:
            runs = [r for r in runs if r["run_id"] in run_ids]

        rows = []
        for r in runs:
            row = {
                "run_id":   r["run_id"],
                "model":    r["model_name"],
                "scenario": r["scenario"],
                **r.get("metrics", {}),
            }
            rows.append(row)
        return pd.DataFrame(rows).sort_values("RMSE") if rows else pd.DataFrame()

    def get_best_run(self, scenario: Optional[str] = None, metric: str = "RMSE") -> Optional[dict]:
        """Return the run with the best metric (lowest RMSE / highest R2)."""
        runs = self._scan_runs()
        if scenario:
            runs = [r for r in runs if r.get("scenario") == scenario]
        runs = [r for r in runs if metric in r.get("metrics", {})]
        if not runs:
            return None
        ascending = metric != "R2"
        return sorted(runs, key=lambda r: r["metrics"][metric], reverse=not ascending)[0]