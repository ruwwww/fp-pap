# Phase 3 LightGBM Experiments

- PACF-selected lags: `[1, 2, 5, 10, 13, 14, 15, 24, 29, 36, 46, 47]`
- Regime split volatility threshold: `0.005261`

## Results
| model             |     rmse |
|:------------------|---------:|
| elasticnet_full   |  369.914 |
| regime_split_safe |  876.062 |
| lgbm_safe         |  889.896 |
| direct_safe       | 1123.21  |
| direct_full       | 1138.52  |
| lgbm_full         | 1604.05  |

## Comparison vs ElasticNet
- ElasticNet RMSE: `369.9137`
- SAFE RMSE: `889.8964` (-140.57% vs ElasticNet)
- FULL RMSE: `1604.0514` (-333.63% vs ElasticNet)
- Direct best RMSE: `1123.2147`
- Regime split RMSE: `876.0621`

## Feature Importance
- SAFE top features: gap_from_trend:2478369.8, dow_cos:2356123.0, usd_lag_1:1829483.0, usd_lag_2:1785125.5, realized_vol_21:1565435.6
- FULL top features: dow_cos:5630603.5, gap_from_trend:5624143.8, VIX_lag1:5399304.4, usd_lag_1:4764131.9, realized_vol_21:3781831.7

## Interpretation
- SAFE answers whether non-linearity helps without OOD exogenous features.
- FULL answers whether OOD exogenous features add lift beyond SAFE.
- Direct multi-step tests whether recursive compounding is the source of error.
- Regime split tests whether volatility clustering is exploitable in a simple gate.
