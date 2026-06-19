# Regime Split Linear

- volatility threshold (median realized vol 21): `0.004993`

| model              |     rmse |
|:-------------------|---------:|
| single_trend_ridge |  327.288 |
| regime_split_ridge | 1435.89  |

## Interpretation
- Single model is the trend baseline.
- Regime split uses the same features but different coefficients under low/high volatility.
- If split beats single, volatility clustering is not just a feature effect; it changes parameterization.
