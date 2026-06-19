# Lag-Llama Backbone Report

Best context length: `128`
Validation RMSE: `506.09`

## OOS Test
RMSE: `607.02`
MAE: `527.06`
MAPE: `3.22%`

## Validation Sweep
| context | val_rmse | val_mae | val_mape |
| --- | ---: | ---: | ---: |
| `32` | `506.34` | `417.33` | `2.72%` |
| `64` | `507.36` | `418.13` | `2.73%` |
| `128` | `506.09` | `417.07` | `2.72%` |

## Benchmark
- Best prior SSM: `RMSE 588.97` (`MarkovAR_4reg_p1`)
- This run uses a local linear trend backbone plus a lag-token transformer residual corrector.
- No direct exogenous predictors were used.
