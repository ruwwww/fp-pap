# Validation Report

## Fold Summary
| scheme     |   ('rmse', 'mean') |   ('rmse', 'std') |   ('mae', 'mean') |   ('mae', 'std') |   ('mape', 'mean') |   ('mape', 'std') |
|:-----------|-------------------:|------------------:|------------------:|-----------------:|-------------------:|------------------:|
| expanding  |            163.205 |           72.4284 |           137.012 |           56.506 |           0.932933 |          0.378216 |
| rolling_5y |            243.735 |          214.523  |           209.04  |          189.84  |           1.4574   |          1.29119  |

Recommended validation for model selection: `expanding`

Note: rolling 5-year windows are still useful as a stress test because they better mimic regime shift, even if average error is higher.