"""
scripts/runs.py
────────────────
CLI tool for inspecting, comparing, and managing experiment artifacts.

Usage:
  # List all runs
  python scripts/runs.py list

  # List last 10 runs
  python scripts/runs.py list --top 10

  # Show detailed info for a run
  python scripts/runs.py show 20260616_103045_xgboost_8020_split

  # Compare specific runs
  python scripts/runs.py compare --runs run1_id,run2_id

  # Compare all runs (best per model per scenario)
  python scripts/runs.py compare

  # Load a model from a past run (prints load command)
  python scripts/runs.py load-model 20260616_103045_xgboost_8020_split

  # Rebuild runs/index.md
  python scripts/runs.py reindex

  # Show best run overall
  python scripts/runs.py best

  # Show best per scenario
  python scripts/runs.py best --scenario "80/20 Split"

  # Delete a run (makes it writable first, then deletes)
  python scripts/runs.py delete 20260616_103045_xgboost_8020_split
"""
import argparse
import json
import os
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.artifacts import ArtifactStore


def cmd_list(store: ArtifactStore, args):
    """List all runs in a table."""
    store.print_index(top_n=args.top)
    index = store.runs_root / "index.md"
    if index.exists():
        print(f"\nFull index: {index}")


def cmd_show(store: ArtifactStore, args):
    """Show detailed information about a single run."""
    run_data = store.load_run(args.run_id)
    if not run_data:
        print(f"Run not found: {args.run_id}")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"Run: {args.run_id}")
    print(f"{'='*65}")

    meta = run_data.get("meta", {})
    print(f"\n[Experiment Info]")
    print(f"  Model    : {meta.get('model_name', '?')}")
    print(f"  Scenario : {meta.get('scenario', '?')}")
    print(f"  Status   : {meta.get('status', '?')}")
    print(f"  Started  : {meta.get('started_at', '?')}")
    print(f"  Elapsed  : {meta.get('elapsed_s', '?')}s")
    print(f"  Target   : {meta.get('target_col', '?')}")

    metrics = run_data.get("metrics", {})
    if metrics:
        print(f"\n[Metrics]")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.6f}")
            else:
                print(f"  {k:12s}: {v}")

    params = run_data.get("params", {})
    if params:
        print(f"\n[Hyperparameters]")
        for k, v in params.items():
            print(f"  {k:25s}: {v}")

    if "predictions" in run_data:
        df = run_data["predictions"]
        print(f"\n[Predictions — {len(df)} rows]")
        print(df.describe().round(2).to_string())

    print(f"\n[Artifact Files]")
    run_dir = Path(run_data["path"])
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(run_dir)
            print(f"  {str(rel):45s} {f.stat().st_size:>10,} bytes")

    print(f"\nPath: {run_data['path']}")
    report = run_dir / "run_report.md"
    if report.exists():
        print(f"Report: {report}")


def cmd_compare(store: ArtifactStore, args):
    """Compare metrics across runs."""
    run_ids = [r.strip() for r in args.runs.split(",")] if args.runs else None
    df = store.compare_runs(run_ids)
    if df.empty:
        print("No runs to compare.")
        return
    print(f"\n{'='*65}")
    print(f"Comparison ({len(df)} runs)")
    print(f"{'='*65}")
    cols = ["run_id", "model", "scenario"] + [
        c for c in ["RMSE", "MAPE", "MAE", "R2"] if c in df.columns
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].sort_values("RMSE" if "RMSE" in cols else cols[0]).to_string(index=False))


def cmd_best(store: ArtifactStore, args):
    """Show the best run by metric."""
    scenario = args.scenario
    metric   = args.metric
    best = store.get_best_run(scenario=scenario, metric=metric)
    if not best:
        print("No completed runs found.")
        return
    print(f"\nBest run ({metric}{'/' + scenario if scenario else ''}):")
    print(f"  Run ID   : {best['run_id']}")
    print(f"  Model    : {best['model_name']}")
    print(f"  Scenario : {best['scenario']}")
    m = best.get("metrics", {})
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.6f}")


def cmd_load_model(store: ArtifactStore, args):
    """Print Python code to load the model from a past run."""
    run_dir = store.runs_root / args.run_id
    model_path = run_dir / "model.joblib"
    if not model_path.exists():
        print(f"No model.joblib in run {args.run_id}")
        sys.exit(1)
    print(f"\n# Load this model in Python:")
    print(f"import joblib")
    print(f"model = joblib.load(r'{model_path}')")
    print(f"\n# Or via ArtifactStore:")
    print(f"from src.artifacts import ArtifactStore")
    print(f"store = ArtifactStore()")
    print(f"model = store.load_model('{args.run_id}')")


def cmd_reindex(store: ArtifactStore, args):
    """Rebuild runs/index.md."""
    path = store.rebuild_index()
    print(f"Index rebuilt: {path}")


def cmd_delete(store: ArtifactStore, args):
    """Delete a run (makes files writable first)."""
    run_dir = store.runs_root / args.run_id
    if not run_dir.exists():
        print(f"Run not found: {args.run_id}")
        sys.exit(1)

    confirm = input(f"Delete run '{args.run_id}'? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    # Make writable
    for f in run_dir.rglob("*"):
        if f.is_file():
            try:
                f.chmod(stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass
    shutil.rmtree(run_dir)
    print(f"Deleted: {run_dir}")
    store.rebuild_index()


def main():
    p = argparse.ArgumentParser(description="EAS — Experiment Artifact Inspector")
    p.add_argument("--config",       default="config/config.yaml")
    p.add_argument("--models-config", default="config/models_config.yaml")
    sub = p.add_subparsers(dest="command")

    # list
    s_list = sub.add_parser("list", help="List all runs")
    s_list.add_argument("--top", type=int, default=50)

    # show
    s_show = sub.add_parser("show", help="Show run details")
    s_show.add_argument("run_id")

    # compare
    s_cmp = sub.add_parser("compare", help="Compare runs")
    s_cmp.add_argument("--runs", default=None, help="Comma-separated run IDs (default: all)")

    # best
    s_best = sub.add_parser("best", help="Show best run")
    s_best.add_argument("--scenario", default=None)
    s_best.add_argument("--metric",   default="RMSE")

    # load-model
    s_load = sub.add_parser("load-model", help="Print Python code to load a model")
    s_load.add_argument("run_id")

    # reindex
    sub.add_parser("reindex", help="Rebuild runs/index.md")

    # delete
    s_del = sub.add_parser("delete", help="Delete a run")
    s_del.add_argument("run_id")

    args = p.parse_args()
    if not args.command:
        p.print_help()
        return

    store = ArtifactStore(
        config_path=args.config,
        models_config_path=args.models_config,
    )

    dispatch = {
        "list":       cmd_list,
        "show":       cmd_show,
        "compare":    cmd_compare,
        "best":       cmd_best,
        "load-model": cmd_load_model,
        "reindex":    cmd_reindex,
        "delete":     cmd_delete,
    }
    dispatch[args.command](store, args)


if __name__ == "__main__":
    main()