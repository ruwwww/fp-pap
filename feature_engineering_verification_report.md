# Feature Engineering Verification

## Walk-forward Summary
| config                               |   ('rmse', 'mean') |   ('rmse', 'std') |   ('mae', 'mean') |   ('mae', 'std') |   ('mape', 'mean') |   ('mape', 'std') |
|:-------------------------------------|-------------------:|------------------:|------------------:|-----------------:|-------------------:|------------------:|
| ar_only                              |            463.813 |           185.162 |           371.437 |         156.802  |            2.50738 |          1.00856  |
| ar_plus_stepwise                     |            635.257 |           113.013 |           543.151 |          90.0047 |            3.69141 |          0.640971 |
| ar_plus_stepwise_market              |            570.757 |           168.517 |           481.573 |         138.533  |            3.25954 |          0.910744 |
| ar_plus_stepwise_market_interactions |            574.153 |           135.02  |           483.462 |          83.5442 |            3.26894 |          0.509102 |

## True OOS Summary
| config                               |    rmse |      mae |     mape |
|:-------------------------------------|--------:|---------:|---------:|
| ar_only                              | 1171.7  |  986.474 |  5.99744 |
| ar_plus_stepwise                     | 1802.49 | 1645.25  | 10.0638  |
| ar_plus_stepwise_market              | 2083.58 | 1831.82  | 11.1714  |
| ar_plus_stepwise_market_interactions | 2133.02 | 1942.33  | 11.8835  |

Best CV config: `ar_only`
Best OOS config: `ar_only`
Best OOS RMSE: `1171.70`

## Interpretation
- Stepwise state features are tested explicitly for BI_rate, CPI, and US_rate.
- Market regime features are tested separately from stepwise features.
- Interactions with VIX are included as gating variables.
- The recursive forecast uses only past target history and known exogenous values at each date.
