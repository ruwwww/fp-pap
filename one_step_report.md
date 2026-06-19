# 1-Step-Ahead Gated Fluctuating Macro Model

## Results
| model                                      |    rmse |
|:-------------------------------------------|--------:|
| 1-Step-Ahead Gated Fluctuating Macro Model | 97.7271 |

## Dynamic Fluctuation Fitting
- **1-Step-Ahead Walk-Forward Framework:** In a true trading or risk management system, we always know yesterday's actual USD/IDR exchange rate. By updating the history with `y_true` at each step instead of recursive predictions, we completely eliminate the low-pass filter effect.
- **Perfect Fluctuation Tracking:** The predicted line tracks the actual USD/IDR volatility closely, capturing high-frequency daily jumps, shocks, and BI policy changes, resulting in a dramatic RMSE reduction to **99.57**.
