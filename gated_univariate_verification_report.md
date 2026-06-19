# Gated Univariate Verification

Context length: `72`
Seed: `5`

| model                            |    rmse |     mae |    mape |
|:---------------------------------|--------:|--------:|--------:|
| univariate_lag_transformer       | 276.635 | 227.456 | 1.41468 |
| univariate_lag_transformer_gated | 277.822 | 228.586 | 1.42149 |

Best config: `univariate_lag_transformer`
Best RMSE: `276.63`

## Gate Rules
- VIX < 15: damp predicted log-return by 80%.
- If US_rate - BI_rate is tight and prediction implies strong Rupiah strengthening, clamp it.
- Clip daily log-return to [-0.015, 0.015].

## Interpretation
- Core model remains univariate log-return only.
- Macro is used only as an external risk governor.
- This verifies whether gating can protect AR inertia without direct exogenous injection.
