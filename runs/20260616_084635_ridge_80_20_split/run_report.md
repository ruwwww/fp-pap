# Run Report: `20260616_084635_ridge_80_20_split`

**Status**: success  
**Started**: 2026-06-16 08:46:35  
**Elapsed**: 0.0s  

## Experiment Info

| Key | Value |
|-----|-------|
| Model | `Ridge` |
| Scenario | 80/20 Split |
| Target | `USDIDR` |
| Run ID | `20260616_084635_ridge_80_20_split` |

## Metrics

| Metric | Value |
|--------|-------|
| RMSE | `453.668568` |
| MAPE | `3.499372` |
| MAE | `88.216977` |
| R2 | `0.252861` |
| SMAPE | `1.219660` |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `fit_time_s` | `0.0046274662017822266` |
| `alpha` | `1.0` |

## Prediction Statistics

| Stat | y_true | y_pred | residual |
|------|--------|--------|----------|
| mean | 8935.92 | 8939.69 | -3.77 |
| std | 524.85 | 260.85 | 453.65 |
| min | 888.11 | 8299.76 | -7963.79 |
| max | 9541.70 | 9464.01 | 500.49 |

## Artifacts

- `config_snapshot.yaml` (1,284 bytes)
- `feature_configs_snapshot.yaml` (2,344 bytes)
- `metrics.json` (148 bytes)
- `model.joblib` (2,975 bytes)
- `models_config_snapshot.yaml` (9,959 bytes)
- `params.json` (57 bytes)
- `predictions.csv` (50,061 bytes)
- `run_metadata.json` (243 bytes)

## Config Snapshot

```yaml
name: Rupiah Resilience — USD/IDR Forecasting
group_number: XX
random_seed: 42
target_column: USDIDR
date_column: Date
feature_columns: ['OIL', 'GOLD', 'SP500', 'IHSG', 'VIX', 'CPI', 'BI_rate', 'US_rate']
```

*Full config: see `config_snapshot.yaml` in this run directory.*
