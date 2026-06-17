# USDIDR Forecasting — ElasticNet Template

> **Mata Kuliah**: Pemodelan dan Analisis Prediktif (PAP)
> **Kaggle Deadline**: Senin, 22 Juni 00:00 WIB
> **Submission Deadline**: Senin, 22 Juni 12:00 WIB

---

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Run (with hyperparameter search)
python elasticnet_forecast_template.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv

# Run (with benchmark against actual test labels)
python elasticnet_forecast_template.py \
    --train_csv data_train.csv \
    --test_csv data_test.csv \
    --submission_csv submission.csv \
    --actual_test_csv submission_real.csv
```

Output is `submission.csv` with columns `Date,USDIDR`.

---

## What It Does

1. Reads train CSV (with `USDIDR` column) and test CSV (features only)
2. Engineers exogenous features (macro rates, spreads, calendar, log-transforms, lags)
3. Builds supervised dataset: predicts **daily log-return** of USDIDR
4. Tunes ElasticNet via `TimeSeriesSplit` cross-validation
5. Recursively forecasts test period (each prediction feeds into the next day's lag features)
6. Saves submission CSV

---

## Target Formulation

The model predicts **log-returns**:

```python
y_t = log(USDIDR_t) - log(USDIDR_{t-1})
```

Reconstructed as:

```python
USDIDR_{t+1} = USDIDR_t * exp(y_pred)
```

This avoids compounding error that plagues level-diff targets in recursive forecasting.

---

## Anti-Leakage Checklist

- [ ] Exogenous features use only current/past values (`shift(n)` where `n >= 1`)
- [ ] Target context features use only past USDIDR history
- [ ] Test USDIDR values are never seen during training or tuning
- [ ] CV uses `TimeSeriesSplit` (no random shuffling)
- [ ] `actual_test_csv` is only used for final RMSE, never fed into the model

---

## Project Structure

```
eas/
├── elasticnet_forecast_template.py   # Single self-contained script
├── data_train.csv                     # Training data (not tracked)
├── data_test.csv                      # Test features (not tracked)
├── submission.csv                     # Generated submission
├── submission_real.csv                # Actual test labels (optional)
├── requirements.txt
└── README.md
```
