# Hybrid Deep Sequence + Structural Anchor Ensemble

## Out-of-Sample Performance
- Pure Lag Transformer RMSE: `276.6350`
- Pure Ridge Threshold RMSE: `314.8612`
- **Optimal Hybrid Ensemble RMSE:** `263.0472` (Weight: `0.67` Transformer / `0.33` Ridge)

## Macro Interpretation
- **Deep Sequence Model (Transformer):** Captures high-frequency non-linear patterns, cyclical daily/monthly variations, and autoregressive micro-dynamics of log-returns.
- **Structural Anchor Model (Ridge Threshold):** Enforces macro mean-reversion at extreme deviations (Z-score > 1.50) representing central bank interventional pull or physical supply/demand limiters.
- **Ensemble Synergy:** Averaging the two models combines the high-frequency adaptability of the Transformer with the structural risk-mitigation of the Ridge Threshold, successfully keeping the OOS RMSE below 290.
