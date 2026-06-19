# SSM Exploration Report

## Benchmark

- Target beat: `RMSE 299` (`LGBM Diff AR-only`)

## 1) Baseline SSM ARIMA Form

`SARIMAX(p,1,0)` on level.

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `SSM_AR10_baseline` | `1274.99` | `1122.90` | `6.84%` |
| `SSM_AR21_baseline` | `1275.29` | `1123.22` | `6.84%` |
| `SSM_AR5_baseline` | `1283.95` | `1132.62` | `6.90%` |

Best baseline order: `AR10`, but still far from benchmark.

## 2) AR Order Sweep

Orders tested: `1, 3, 5, 10, 21`.

Best order by RMSE: `10`.

Plot: `ssm_ar_order_rmse.png`

## 3) TVP-AR

`SARIMAX(time_varying_regression=True)` on diff space with lagged diff regressors.

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `TVP_AR10` | `1277.06` | `1125.15` | `6.85%` |
| `TVP_AR5` | `1281.45` | `1129.89` | `6.88%` |
| `TVP_AR1` | `1284.57` | `1133.29` | `6.90%` |

Conclusion:

- coefficients do move over time,
- but the TVP formulation did **not** improve OOS.

Coefficient summary for `TVP_AR10`:

- `beta.lag1` std `0.60`, p95 abs change `1.00`
- `beta.lag4` std `0.90`, p95 abs change `1.38`
- several other lags also drift materially

Plot: `tvp_ar10_coefficients.png`

## 4) Structural Break SSM

Fit only on latest pre-test regime (`2022-03` to `2023-05`).

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `Break_2022_2023_AR10` | `1264.15` | `1111.24` | `6.77%` |
| `Break_2022_2023_AR21` | `1279.85` | `1128.13` | `6.87%` |
| `Break_2022_2023_AR5` | `1282.44` | `1130.97` | `6.89%` |

This is the best fixed-break variant, but still nowhere near 299.

## 5) Regime-Switching SSM

`MarkovAutoregression` with switching AR coefficients.

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `MarkovAR_4reg_p1` | `588.97` | `510.49` | `3.12%` |
| `MarkovAR_2reg_p1` | `664.63` | `578.80` | `3.53%` |
| `MarkovAR_2reg_p3` | `2865.97` | `2511.42` | `15.30%` |

This is the strongest SSM family result.

Interpretation:

- regime switching helps a lot vs fixed AR SSM,
- but still does not beat the `299` benchmark,
- and the model still worsens materially by 2026.

## 6) Ensemble

Simple SSM ensembles did not help.

Best ensemble observed:

- `Avg(SSM_AR10, MarkovAR_4reg_p1)` RMSE `989.25`

## 7) Per-Year Robustness

Best SSM variant: `MarkovAR_4reg_p1`

| year | RMSE |
| --- | ---: |
| 2023 | `290.56` |
| 2024 | `450.35` |
| 2025 | `387.19` |
| 2026 | `583.70` |

So the best SSM is competitive early, then degrades.

## Final Verdict

SSM did help once the formulation respected the data:

- diff space,
- AR-dominant structure,
- regime sensitivity.

But even the best tested SSM variant did **not** beat `RMSE 299`.

Practical conclusion:

- the most compatible SSM here is a regime-switching AR in diff space,
- yet gradient boosting still remains stronger on this horizon.
