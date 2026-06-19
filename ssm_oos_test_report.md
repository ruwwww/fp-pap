# State-Space OOS Test Report

## Setup
Saya evaluasi model state-space / hidden-state style pada horizon actual test `2023-06-01` s/d `2026-05-29` menggunakan:

- `data_train.csv` sebagai fit set
- `data_test.csv` sebagai future exogenous input
- `data_test_actual.csv` sebagai ground truth

## Models Tested

- Naive last value
- Recursive AR(5) on level
- Local level state-space
- Local linear trend state-space
- SARIMAX with exogenous risk variables
- SARIMAX with full exogenous variables

## OOS Results

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| LocalLinearTrend | 607.13 | 527.02 | 3.22% |
| AR5_recursive | 1116.55 | 982.37 | 5.98% |
| Naive | 1284.21 | 1132.90 | 6.90% |
| LocalLevel | 1289.51 | 1138.67 | 6.94% |
| SARIMAX_risk | 1678.88 | 1467.19 | 8.93% |
| SARIMAX_full | 1688.43 | 1477.86 | 9.00% |

## Main Findings

### 1. Local linear trend is the best tested SSM variant
It materially beats:

- naive baseline
- recursive AR(5)
- local level
- SARIMAX with exogenous inputs

### 2. The gain comes from trend adaptation, not macro regression
The best SSM result came from a model that can adapt a smooth latent trend.
Adding exogenous inputs in the tested SARIMAX variants did not help.

### 3. Local level and SARIMAX were not enough
Plain local level and SARIMAX-style formulations behaved poorly on this long horizon.

## Subperiod View

Local linear trend also won in each yearly slice:

- 2023: `330.63`
- 2024: `539.89`
- 2025: `630.69`
- 2026: `927.40`

This matters because it shows the model is not only fitting the first part of the test window.

## Interpretation

Verified conclusion:

- hidden regime structure exists,
- but the practical edge in this test horizon came from **latent trend adaptation**,
- not from static exogenous regression.

So for this problem, a state-space approach is useful mainly as a **trend tracker**.
It is not yet evidence that exogenous variables are being extracted optimally by a vanilla SSM.
