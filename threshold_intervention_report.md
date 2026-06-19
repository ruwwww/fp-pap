# Threshold Intervention Experiment

- validation baseline RMSE: `411.2950`
- selected threshold: `1.50`
- validation best RMSE: `322.0744`

| model                         |    rmse |
|:------------------------------|--------:|
| threshold_mean_reversion_1.50 | 314.861 |
| baseline_trend_ridge          | 327.288 |

## Interpretation
- intervention_pull is only active in extreme positive deviations.
- threshold is chosen on an internal validation split, not on the test labels.
- if this wins, mean reversion is asymmetric rather than smooth.
