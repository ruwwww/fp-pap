import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, load_test
from model_a.model_xgb import XGBoostPAP
from model_a.features_xgb import build_features_xgb

TARGET = "USDIDR"
DATE = "Date"
SUBMISSION_DIR = Path("data/raw")


def forecast():
    print("=== Model A — XGBoost Kaggle Submission Forecast ===\n")

    train_df = load_train()
    test_df  = load_test()
    submission = pd.read_csv(SUBMISSION_DIR / "submission.csv")
    submission[DATE] = pd.to_datetime(submission[DATE])

    print(f"Train: {len(train_df)} rows ({train_df[DATE].min().date()} -> {train_df[DATE].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df[DATE].min().date()} -> {test_df[DATE].max().date()})")

    # ---------- Train ----------
    # Use last 20% of train as validation for early stopping
    n_val = max(1, int(len(train_df) * 0.15))
    train_fit = train_df.iloc[:-n_val].copy()
    train_val  = train_df.iloc[-n_val:].copy()

    full_tr = pd.concat([train_fit, train_val], ignore_index=True)
    X_full, y_full = build_features_xgb(full_tr)

    n_nan = len(full_tr) - len(X_full)
    n_fit = len(train_fit) - n_nan

    X_tr  = X_full.iloc[:n_fit]
    y_tr  = y_full.iloc[:n_fit]
    X_val = X_full.iloc[n_fit:]
    y_val = y_full.iloc[n_fit:]

    print(f"Fit rows: {len(X_tr)}  Val rows: {len(X_val)}  Features: {X_tr.shape[1]}")

    model = XGBoostPAP(
        n_estimators=2000, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    print("Model trained.")

    # ---------- Recursive forecast ----------
    preds   = []
    history = train_df.copy()

    for i, row in test_df.iterrows():
        # Append the new test row (target unknown)
        new_row = row.to_dict()
        new_row[TARGET] = np.nan
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        # Build features on a recent window
        recent = history.iloc[-100:]
        X_rec, _ = build_features_xgb(recent)

        if len(X_rec) > 0:
            pred = model.predict(X_rec.iloc[[-1]])[0]
        else:
            pred = history[TARGET].dropna().iloc[-1]

        # Clip to a sane range (prevents wild extrapolation)
        pred = float(np.clip(pred, 12000, 20000))

        preds.append(pred)
        history.loc[history.index[-1], TARGET] = pred

        if (len(preds)) % 100 == 0:
            print(f"  Predicted {len(preds)}/{len(test_df)} rows...")

    y_pred = np.array(preds)
    print(f"\nPrediction range: {y_pred.min():.0f} - {y_pred.max():.0f}")
    print(f"Prediction mean:  {y_pred.mean():.0f}")

    sub = pd.DataFrame({DATE: submission[DATE].values, TARGET: y_pred})
    out_path = SUBMISSION_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"Submission saved -> {out_path} ({len(sub)} rows)")
    return sub


if __name__ == "__main__":
    forecast()
