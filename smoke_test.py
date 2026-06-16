"""
verify_dataset.py - dataset-specific verification
Run: python verify_dataset.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.features.feature_engineering import TimeSeriesFeatureEngineer
from src.models.ml_models import XGBoostModel
from src.evaluation.metrics import evaluate_all

print("[1] Loading data...")
loader   = DataLoader()
train_df = loader.load_train()
test_df  = loader.load_kaggle_test()
sub_tmpl = loader.load_submission_template()
print(f"    Train : {train_df.shape}  ({train_df['Date'].min().date()} -> {train_df['Date'].max().date()})")
print(f"    Test  : {test_df.shape}   ({test_df['Date'].min().date()}  -> {test_df['Date'].max().date()})")
print(f"    Sub   : {sub_tmpl.shape}")
print(f"    Train cols: {list(train_df.columns)}")
print(f"    Test  cols: {list(test_df.columns)}")

print("\n[2] Internal split 80/20...")
tr_80, te_20 = loader.get_internal_split(train_df, 0.80)

print("\n[3] Feature engineering (no leakage)...")
feat_eng = TimeSeriesFeatureEngineer(target_col="USDIDR", date_col="Date")
tr_fe    = feat_eng.fit_transform(tr_80).dropna()
full_ctx = pd.concat([tr_80, te_20], ignore_index=True)
te_fe    = feat_eng.transform(te_20, full_ctx).dropna()

feat_cols = feat_eng.get_feature_columns(tr_fe)
X_tr = tr_fe[feat_cols]
y_tr = tr_fe["USDIDR"]
X_te = te_fe[feat_cols]
y_te = te_fe["USDIDR"]

print(f"    Features  : {len(feat_cols)}")
print(f"    X_train   : {X_tr.shape}")
print(f"    X_test    : {X_te.shape}")
print(f"    USDIDR range: {y_tr.min():.0f} - {y_tr.max():.0f}")
print(f"    First 5 features : {feat_cols[:5]}")
print(f"    Last  5 features : {feat_cols[-5:]}")

print("\n[4] Anti-leakage check...")
assert "usdidr_lag_1" in feat_cols, "lag_1 missing!"
# Verify shift(1) pattern: lag_1[i] == usdidr[i-1] using the engineered df
# (use tr_fe directly which has reset index after dropna)
tr_fe_check = feat_eng.fit_transform(tr_80)  # no dropna — keeps index aligned
lag1 = tr_fe_check["usdidr_lag_1"].values
actual = tr_fe_check["USDIDR"].values
ok = True
for i in range(5, 20):
    if not np.isnan(lag1[i]) and not np.isnan(actual[i - 1]):
        if abs(lag1[i] - actual[i - 1]) > 1:
            print(f"    LEAK DETECTED at i={i}: expected={actual[i-1]:.0f} got={lag1[i]:.0f}")
            ok = False
            break
print(f"    Lag-1 leakage check: {'PASS' if ok else 'FAIL'}")

print("\n[5] Quick model test (XGBoost 50 trees)...")
model = XGBoostModel(params={"n_estimators": 50, "random_state": 42})
model.fit_timed(X_tr, y_tr)
preds = model.predict(X_te)
m = evaluate_all(y_te.values, preds)
print(f"    Fit time : {model.fit_time_:.2f}s")
print(f"    RMSE     : {m['RMSE']:.2f} IDR")
print(f"    MAPE     : {m['MAPE']:.3f} %")
print(f"    MAE      : {m['MAE']:.2f} IDR")
print(f"    R2       : {m['R2']:.4f}")

print("\n[6] 3 scenarios check...")
for label, ratio in [("80/20", 0.80), ("70/30", 0.70), ("60/40", 0.60)]:
    tr, te = loader.get_internal_split(train_df, ratio)
    print(f"    {label}: train={len(tr)} test={len(te)} "
          f"({tr['Date'].min().date()} - {tr['Date'].max().date()}) | "
          f"({te['Date'].min().date()} - {te['Date'].max().date()})")

print("\n[7] Submission template check...")
print(f"    Dates   : {sub_tmpl['Date'].min()} -> {sub_tmpl['Date'].max()}")
print(f"    Rows    : {len(sub_tmpl)}")
print(f"    Columns : {list(sub_tmpl.columns)}")

print("\n" + "="*55)
print("DATASET READY FOR TRAINING")
print("="*55)
print("\nRun training:")
print("  python scripts/run_all.py")
print("\nRun quick ML only:")
print("  python scripts/train.py --skip-category DL --scenarios 80_20")
print("\nRun search first:")
print("  python scripts/search.py --models xgboost,lightgbm --trials 50")
