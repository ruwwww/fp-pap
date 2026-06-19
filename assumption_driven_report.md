# Assumption-Driven USDIDR Experiment

## Phase 0
- ADF/KPSS level: p=0.7158 / 0.01
- ADF/KPSS diff: p=9.758e-21 / 0.1
- ADF/KPSS log-return: p=4.405e-20 / 0.1
- Selected AR order from PACF: `47`
- Ljung-Box squared returns p-values: lag10=5.297e-217, lag20=3.965e-287, lag60=0
- CUSUM statistic: 0.7831, p=0.5718

## Phase 1 Baselines
| model            |     rmse |   directional_hit_rate |
|:-----------------|---------:|-----------------------:|
| naive_last       | 1284.21  |               0.772201 |
| naive_drift      |  606.009 |              54.1828   |
| seasonal_naive_5 | 1300.51  |              53.0245   |

## Phase 2 Linear Models
| model                 |    rmse |   directional_hit_rate |
|:----------------------|--------:|-----------------------:|
| ar_p                  | 549.876 |                54.0541 |
| ar_plus_trend         | 434.033 |                53.0245 |
| ar_plus_verified_exog | 495.997 |                51.9949 |
| elasticnet_full       | 369.914 |                54.4402 |

## Best Model
- model: `elasticnet_full`
- RMSE: `369.9137`
- Improvement over naive_last: `71.20%`
- Selected exogenous features: `SP500, VIX`

## Per-Year RMSE
|   year |     rmse | model                 |
|-------:|---------:|:----------------------|
|   2023 |  453.795 | naive_last            |
|   2024 |  925.113 | naive_last            |
|   2025 | 1485.92  | naive_last            |
|   2026 | 2078.41  | naive_last            |
|   2023 |  306.574 | ar_p                  |
|   2024 |  501.836 | ar_p                  |
|   2025 |  558.892 | ar_p                  |
|   2026 |  837.685 | ar_p                  |
|   2023 |  243.14  | ar_plus_trend         |
|   2024 |  429.756 | ar_plus_trend         |
|   2025 |  473.903 | ar_plus_trend         |
|   2026 |  542.932 | ar_plus_trend         |
|   2023 |  303.966 | ar_plus_verified_exog |
|   2024 |  473.56  | ar_plus_verified_exog |
|   2025 |  489.586 | ar_plus_verified_exog |
|   2026 |  731.738 | ar_plus_verified_exog |
|   2023 |  209.062 | elasticnet_full       |
|   2024 |  448.179 | elasticnet_full       |
|   2025 |  369.738 | elasticnet_full       |
|   2026 |  335.317 | elasticnet_full       |

## OOD Shift
| feature   |   train_mean |   test_mean |   mean_shift_sigma |   train_std |   wasserstein |   test_gt_2sigma_pct |
|:----------|-------------:|------------:|-------------------:|------------:|--------------:|---------------------:|
| OIL       |     71.42    |    73.714   |          0.102647  |    22.3483  |     11.1496   |             0        |
| GOLD      |   1464.11    |  2978.18    |          5.83997   |   259.26    |   1514.07     |            84.8329   |
| SP500     |   2452.26    |  5706.38    |          3.20789   |  1014.41    |   3254.12     |            88.3033   |
| IHSG      |   5195.94    |  7308.66    |          1.86921   |  1130.28    |   2112.73     |            29.3059   |
| VIX       |     18.8039  |    17.1042  |         -0.21522   |     7.89768 |      2.28413  |             0.771208 |
| CPI       |      4.22674 |     2.18725 |         -1.32253   |     1.54211 |      2.03949  |            33.162    |
| BI_rate   |      5.63601 |     5.61183 |         -0.0183193 |     1.31998 |      0.681172 |             0        |
| US_rate   |      0.76231 |     4.65325 |          3.54713   |     1.09693 |      3.89094  |           100        |

## Verdict
- If the AR family does not beat naive meaningfully, the sample is effectively AR-ceilinged.
- Exogenous features are only kept when they improve OOS on the holdout, not because they are plausible in theory.
- SSM is not warranted unless a later phase beats these linear baselines with stable per-year gains.
