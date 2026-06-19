# OOS Exogenous Performance Report

## Setup

- Train: `data_train.csv`
- Future exog: `data_test.csv`
- Ground truth: `data_test_actual.csv`
- Scope: only the exogenous variables that were just investigated with Granger
  - `SP500`
  - `VIX`
  - `IHSG`
  - `GOLD`
  - `OIL`
  - `US_rate`

## Best OOS Results

### Ridge, recursive forecast on level

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `Diff_AR_only` | `573.72` | `495.89` | `3.03%` |
| `Diff_AR_exog_diff` | `1145.13` | `964.05` | `5.85%` |
| `Level_AR_only` | `1205.30` | `1062.43` | `6.47%` |
| `Naive` | `1284.21` | `1132.90` | `6.90%` |

### LightGBM, recursive forecast on diff

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `LGBM_Diff_AR_only` | `299.25` | `238.19` | `1.48%` |
| `LGBM_Diff_AR_exog_diff` | `1879.91` | `1360.38` | `8.19%` |
| `LGBM_Diff_AR_exog_both` | `7905.71` | `6350.33` | `38.56%` |

## Main Finding

Di actual OOS horizon, exogenous variables yang tadi signifikan di Granger **tidak improve forecast**.

Malah:

- AR-only diff model paling stabil,
- menambahkan exog lag terpilih membuat performa turun,
- efek ini konsisten di Ridge dan LightGBM.

## Interpretation

Ini berarti:

1. Sinyal Granger ada di train, tapi tidak robust enough untuk OOS langsung.
2. Exog yang dipilih kemungkinan menangkap common shock / regime coupling, bukan signal yang stabil untuk forecast point-wise.
3. Model yang paling berguna untuk horizon ini tetap AR-dominant.

## Practical Conclusion

Kalau targetnya murni performa forecast di `data_test_actual.csv`, maka prioritasnya:

1. AR / diff-AR backbone
2. regime-aware adaptation
3. exog only if gated or time-varying

Jadi exog ini layak dilihat sebagai **regime cue**, bukan direct predictor default.
