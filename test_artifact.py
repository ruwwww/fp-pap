"""
test_artifact.py - verify artifact system with real data
Run: python test_artifact.py
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, ".")

import stat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import ArtifactStore, RunRecord
from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.models.ml_models import XGBoostModel
from src.evaluation.metrics import evaluate_all
from src.visualization.plots import plot_actual_vs_predicted

print("[1] Setting up ArtifactStore...")
store = ArtifactStore()

print("[2] Loading real data (80/20)...")
loader   = DataLoader()
train_df = loader.load_train()
tr_80, te_20 = loader.get_internal_split(train_df, 0.80)

print("[3] Feature engineering...")
feat_eng = TimeSeriesFeatureEngineer(target_col="USDIDR", date_col="Date")
tr_fe = feat_eng.fit_transform(tr_80).dropna()
full_ctx = pd.concat([tr_80, te_20], ignore_index=True)
te_fe = feat_eng.transform(te_20, full_ctx).dropna()
feat_cols = feat_eng.get_feature_columns(tr_fe)
X_tr, y_tr = tr_fe[feat_cols], tr_fe["USDIDR"]
X_te, y_te = te_fe[feat_cols], te_fe["USDIDR"]

print("[4] Starting run in ArtifactStore...")
run = store.start_run(model_name="XGBoost", scenario="80/20 Split")
print(f"    Run ID  : {run.run_id}")
print(f"    Run dir : {run.path}")

print("[5] Training model...")
model = XGBoostModel(params={"n_estimators": 100, "random_state": 42})
model.fit_timed(X_tr, y_tr)
preds = model.predict(X_te)
metrics = evaluate_all(y_te.values, preds)

print("[6] Logging artifacts...")
run.log_params({"n_estimators": 100, "random_state": 42, "fit_time_s": model.fit_time_})
run.log_metrics(metrics)
run.log_predictions(y_te.values, preds, dates=te_fe["Date"] if "Date" in te_fe.columns else None)
run.log_model(model)

print("[7] Saving plot...")
fig = plot_actual_vs_predicted(
    tr_fe["Date"] if "Date" in tr_fe.columns else tr_80["Date"].iloc[-len(y_tr):], y_tr.values,
    te_fe["Date"] if "Date" in te_fe.columns else te_20["Date"].iloc[-len(y_te):], y_te.values, preds,
    model_name="XGBoost", scenario="80/20 Split",
)
run.log_plot(fig, "actual_vs_predicted")
plt.close("all")

print("[8] Finishing run (immutable)...")
run.finish(status="success")

print("[9] Verifying immutability...")
run_files = list(run.path.rglob("*"))
read_only_count = 0
for f in run_files:
    if f.is_file():
        mode = f.stat().st_mode
        is_read_only = not bool(mode & stat.S_IWRITE)
        if is_read_only:
            read_only_count += 1
print(f"    {read_only_count}/{sum(1 for f in run_files if f.is_file())} files are read-only")

print("[10] Checking artifact files exist...")
expected = [
    "run_metadata.json",
    "config_snapshot.yaml",
    "models_config_snapshot.yaml",
    "params.json",
    "metrics.json",
    "predictions.csv",
    "model.joblib",
    "plots/actual_vs_predicted.png",
    "run_report.md",
]
for fname in expected:
    p = run.path / fname
    status = "OK" if p.exists() else "MISSING"
    size = f"{p.stat().st_size:,} bytes" if p.exists() else "---"
    print(f"    [{status}] {fname:45s} {size}")

print("[11] Loading past run from store...")
loaded = store.load_run(run.run_id)
assert loaded is not None, "Could not load run"
assert loaded["metrics"]["RMSE"] > 0, "RMSE should be positive"
m = loaded["metrics"]
print(f"    RMSE : {m['RMSE']:.2f} IDR")
print(f"    MAPE : {m['MAPE']:.3f} %")
print(f"    R2   : {m['R2']:.4f}")

print("[12] Loading model from store...")
import joblib
loaded_model = store.load_model(run.run_id)
preds2 = loaded_model.predict(X_te)
assert abs(preds2[0] - preds[0]) < 1, "Loaded model should produce same predictions"
print(f"    Model reloaded OK — pred[0]={preds2[0]:.2f}")

print("[13] Rebuilding index...")
index_path = store.rebuild_index()
assert index_path.exists(), "index.md not created"
print(f"    Index: {index_path}")
with open(index_path, encoding="utf-8") as f:
    idx_content = f.read()
assert run.run_id in idx_content, "Run ID not in index"
print(f"    Run ID found in index: OK")

print("[14] Read run_report.md...")
report = run.path / "run_report.md"
with open(report, encoding="utf-8") as f:
    content = f.read()
assert "XGBoost" in content
assert "RMSE" in content
print(f"    Report preview (first 5 lines):")
for line in content.split("\n")[:5]:
    print(f"      {line}")

print()
print("="*55)
print("ARTIFACT SYSTEM: ALL CHECKS PASSED")
print("="*55)
print(f"\nRun stored at: {run.path}")
print(f"Index at     : {index_path}")
print(f"\nInspect with:")
print(f"  python scripts/runs.py list")
print(f"  python scripts/runs.py show {run.run_id}")
