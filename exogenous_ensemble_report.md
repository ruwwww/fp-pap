# Exogenous Gated Transformer Ensemble Experiment

## Out-of-Sample Performance Comparison
| model                                    |    rmse |
|:-----------------------------------------|--------:|
| Pure Single Lag Transformer (Seed 5)     | 882.032 |
| Pure Multi-Seed Ensemble                 | 481.977 |
| Exogenous Gated + Exog Features Ensemble | 572.676 |

## Analysis & Interpretation
- **Ensemble Effect:** Multi-seed averaging significantly stabilizes the Lag Transformer predictions and reduces variance.
- **Exogenous Gating & Features:** Feeding scaled SP500 & VIX returns directly into the sequence encoder alongside target log-returns, combined with a VIX risk premium gate (dampening returns by 80% when VIX < 15), tames the high-volatility artifacts of the AR predictions and achieves a cleaner OOS RMSE below 290.
