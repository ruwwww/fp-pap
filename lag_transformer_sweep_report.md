# Lag Transformer Sweep

ElasticNet baseline RMSE: `386.36`

| model | context | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: | ---: |
| `pure_lag_transformer_return` | `64` | `294.91` | `227.44` | `1.41%` |
| `elasticnet_plus_lag_residual` | `128` | `386.36` | `314.97` | `1.94%` |
| `elasticnet_plus_lag_residual` | `64` | `386.36` | `314.97` | `1.94%` |
| `elasticnet_plus_lag_residual` | `32` | `386.36` | `314.98` | `1.94%` |
| `pure_lag_transformer_return` | `128` | `798.49` | `629.61` | `3.85%` |

Best result is the first row of the table.
