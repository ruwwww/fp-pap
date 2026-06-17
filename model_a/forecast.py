import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import load_train, load_test
from model_a.model import ElasticNetPAC
from model_a.features import build_features

TARGET = "USDIDR"
DATE = "Date"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SUBMISSION_DIR = Path("data/raw")


def forecast():
    print("=== Model A — Kaggle Submission Forecast ===\n")

    train_df = load_train()
    test_df = load_test()
    submission = pd.read_csv(SUBMISSION_DIR / "submission.csv")
    submission[DATE] = pd.to_datetime(submission[DATE])

    print(f"Train: {len(train_df)} rows ({train_df[DATE].min().date()} -> {train_df[DATE].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df[DATE].min().date()} -> {test_df[DATE].max().date()})")

    train_feat, y_train = build_features(train_df)
    model = ElasticNetPAC(alpha=1.0, l1_ratio=0.5)
    model.fit(train_feat, y_train)
    print(f"Model trained on {len(train_feat)} rows, {train_feat.shape[1]} features")

    preds = []
    history = train_df.copy()

    for i, row in test_df.iterrows():
        new_row = row.to_dict()
        new_row[TARGET] = np.nan
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        recent = history.iloc[-70:]
        X_recent, _ = build_features(recent)
        if len(X_recent) > 0:
            pred = model.predict(X_recent.iloc[[-1]])[0]
        else:
            pred = history[TARGET].iloc[-1]

        preds.append(pred)
        history.loc[history.index[-1], TARGET] = pred

        if (i + 1) % 100 == 0:
            print(f"  Predicted {i + 1}/{len(test_df)} rows...")

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