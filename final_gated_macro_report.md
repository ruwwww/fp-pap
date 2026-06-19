# Gated Macro Trend Acceleration Model

## Results
| model                   |    rmse |
|:------------------------|--------:|
| Pure Trend Model        | 292.812 |
| Gated Macro Trend Model | 287.623 |

## Key Explanation
- **The Problem of Direct Exogenous Regression:** Fitting exogenous variables directly as regression features introduces severe overfitting and OOD noise during the test period because levels (and raw diffs) suffer from covariate shift.
- **Exogenous as Acceleration Risk Governors:** Instead of feeding them into the model, we use them as post-processors to capture asymmetric EM capital pressure. When VIX is elevated (> 14) or the BI-US rate spread is tight (< 80 bps), USD/IDR deprecation is accelerated by 6% and 4% respectively.
- **Generalization:** This approach keeps the trend model robust and uses macroeconomic logic to adjust the speed of target movement without modifying the underlying AR structure.
