# Exogenous EDA Report

## 1. Target Inertia
- ADF (level) p-value: 0.715823
- KPSS (level) p-value: 0.010000
- ADF (log return) p-value: 0.000000
- KPSS (log return) p-value: 0.100000
- Hurst exponent on USDIDR log return: 0.5175

### ACF/PACF Notes
- Top ACF lags: 1(-0.315), 5(0.083), 85(0.082), 10(0.075), 15(0.073), 89(-0.067), 135(0.065), 86(-0.064), 30(0.061), 72(0.060)
- Top PACF lags: 1(-0.315), 2(-0.109), 5(0.068), 10(0.064), 85(0.062), 15(0.049), 89(-0.045), 29(-0.044), 111(0.043), 71(-0.041)

## 2. Exogenous Mixed-Frequency Audit
|   max_run |   mean_run |   p95_run |   changes |   stale_days_pct | feature   |   train_adf_p |   train_kpss_p |
|----------:|-----------:|----------:|----------:|-----------------:|:----------|--------------:|---------------:|
|         3 |    1.04045 |      1    |      3361 |          3.88905 | OIL       |   0.282202    |           0.01 |
|         3 |    1.04138 |      1    |      3358 |          3.97484 | GOLD      |   0.65579     |           0.01 |
|         3 |    1.03675 |      1    |      3373 |          3.5459  | SP500     |   0.923217    |           0.01 |
|         8 |    1.07005 |      1    |      3268 |          6.54847 | IHSG      |   0.296676    |           0.01 |
|         3 |    1.04076 |      1    |      3360 |          3.91764 | VIX       |   1.32057e-09 |           0.01 |
|       262 |  249.857   |    261.35 |        13 |         99.6283  | CPI       |   0.489436    |           0.01 |
|       393 |   87.45    |    319.5  |        39 |         98.8848  | BI_rate   |   0.556464    |           0.01 |
|       129 |   29.6441  |     66    |       117 |         96.6543  | US_rate   |   0.963608    |           0.01 |

## 3. Low-Frequency Event / Stale Behavior
|   change_days |   mean_abs_usdret_on_change |   mean_abs_usdret_no_change |   corr_change_flag_abs_usdret |   mean_delta |   median_delta | feature   |
|--------------:|----------------------------:|----------------------------:|------------------------------:|-------------:|---------------:|:----------|
|            13 |                  0.00736323 |                  0.00437817 |                    0.0350161  |   -0.112678  |      -0.167763 | CPI       |
|            39 |                  0.00702254 |                  0.00435956 |                    0.0539035  |   -0.0192308 |      -0.25     | BI_rate   |
|           117 |                  0.00463065 |                  0.00438091 |                    0.00865676 |    0.0423077 |       0.01     | US_rate   |

## 4. Train-Test Scale Shift
|   wasserstein |   kl_divergence |   train_mean |   test_mean |   train_p90 |   test_p90 |   mean_ratio |   p90_ratio | feature   |
|--------------:|----------------:|-------------:|------------:|------------:|-----------:|-------------:|------------:|:----------|
|     11.1496   |         7.064   |     71.42    |    73.714   |   101.28    |   87.519   |     1.03212  |    0.864129 | OIL       |
|   1514.07     |        20.4072  |   1464.11    |  2978.18    |  1835.99    | 4588.4     |     2.03412  |    2.49914  | GOLD      |
|   3254.12     |        21.5967  |   2452.26    |  5706.38    |  4108.99    | 6875.3     |     2.32699  |    1.67323  | SP500     |
|   2112.73     |        16.0605  |   5195.94    |  7308.66    |  6656.28    | 8228.7     |     1.40661  |    1.23623  | IHSG      |
|      2.28413  |         1.76918 |     18.8039  |    17.1042  |    27.666   |   21.945   |     0.909607 |    0.793212 | VIX       |
|      2.03949  |        22.3132  |      4.22674 |     2.18725 |     6.39493 |    3.66939 |     0.51748  |    0.573796 | CPI       |
|      0.681172 |        14.5402  |      5.63601 |     5.61183 |     7.5     |    6.25    |     0.99571  |    0.833333 | BI_rate   |
|      3.89094  |        24.6164  |      0.76231 |     4.65325 |     2.33    |    5.33    |     6.10415  |    2.28755  | US_rate   |

## 5. Conditional Volatility Slices
|   vix_quartile | feature      |     pearson |    spearman |   n |
|---------------:|:-------------|------------:|------------:|----:|
|              0 | sp500_ret_l1 | -0.106608   | -0.113811   | 875 |
|              0 | vix_l1       |  0.00620809 |  0.00910069 | 875 |
|              0 | ihsg_ret_l1  | -0.141399   | -0.0991823  | 875 |
|              1 | sp500_ret_l1 | -0.132692   | -0.166399   | 874 |
|              1 | vix_l1       |  0.00534359 |  0.00230393 | 874 |
|              1 | ihsg_ret_l1  | -0.117524   | -0.123915   | 874 |
|              2 | sp500_ret_l1 | -0.166962   | -0.212074   | 876 |
|              2 | vix_l1       |  0.00128786 | -0.0297964  | 876 |
|              2 | ihsg_ret_l1  | -0.159031   | -0.17388    | 876 |
|              3 | sp500_ret_l1 | -0.231438   | -0.180993   | 872 |
|              3 | vix_l1       |  0.112254   |  0.0224062  | 872 |
|              3 | ihsg_ret_l1  | -0.15314    | -0.133542   | 872 |

### Directional AUC by VIX Quartile
|   vix_quartile |      auc |   train_n |   test_n |
|---------------:|---------:|----------:|---------:|
|              0 | 0.436707 |       612 |      263 |
|              1 | 0.530545 |       611 |      263 |
|              2 | 0.548184 |       613 |      263 |
|              3 | 0.532993 |       610 |      262 |

## 6. Structural Breaks
|   f_stat |   p_value |   n1 |   n2 | break_date   |
|---------:|----------:|-----:|-----:|:-------------|
|  2.34186 | 0.0962993 | 2237 | 1260 | 2018-08-01   |
|  2.73269 | 0.0651834 | 2649 |  848 | 2020-03-01   |

### Rolling OLS Summary
- Rolling OLS window: 252
- SP500 coef mean: -0.106410
- SP500 coef min/max: -0.289309 / 0.027973
- Percent positive: 4.92%
- Sign flips: 26

## Bottom Line
- Exogenous variables are not noisy corruption; they are mixed-frequency, stale, and regime-sensitive.
- Risk proxies (SP500, VIX, IHSG) are the most informative slices.
- Low-frequency rates need state/change features, not raw levels only.
- There is clear train-test scale shift, especially for rate and equity variables.
- Before seed-variance work, the right next step is to use exogenous as regime cues and hidden-state features, not direct daily continuous predictors.
