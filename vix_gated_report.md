# VIX Gated Regime Switching Experiment

## Validation Phase
- Validation Baseline (always trend) RMSE: `411.2950`
- Validation Best Gated RMSE: `319.9216`
- Selected VIX Gate Threshold: `19.5`

## Out-of-Sample Evaluation
| model                          |    rmse |
|:-------------------------------|--------:|
| Pure Trend Model               | 327.288 |
| Pure Threshold Model (thr=1.5) | 314.861 |
| VIX Gated Model (gate=19.5)    | 332.654 |

## Key Findings
- Testing the hypothesis that high-VIX environments (VIX_lag1 > 19.5) require threshold mean-reversion modeling, while low-VIX environments are trend-dominated.
- Gating selection was done entirely on the training validation fold to prevent leakage.
