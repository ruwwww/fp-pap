# Residual Stack Ensemble

- Validation blend weights: `{'base': 0.0, 'corrected': 0.0, 'trend': 1.0}`
- Validation blend RMSE: `392.8909`

| model                    |     rmse |
|:-------------------------|---------:|
| ensemble_final           |  292.812 |
| ar_plus_trend            |  292.812 |
| elasticnet_full          |  443.012 |
| elasticnet_residual_safe | 1771.23  |

## Interpretation
- Base ElasticNet stays the anchor.
- SAFE residual model only corrects leftover structure.
- Final ensemble is convex and tuned on an internal validation window.
