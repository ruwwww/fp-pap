# Gated Fluctuating Macro Model (Confidence-Boosted)

## Results
| model                         |    rmse |
|:------------------------------|--------:|
| Static Pure Trend Model       | 294.825 |
| Gated Fluctuating Macro Model | 269.376 |

## Core Quant Breakthrough
- **The Problem of Smooth Trend Extrapolations:** Autoregressive models in recursive multi-step forecasting act as low-pass filters, smoothing out all daily volatility and producing a flat/smooth line.
- **Exogenous as Acceleration Risk Governors (Confidence-Boosted):** We separate the trend model (Ridge, alpha=1.0 on target lags) from the high-frequency daily shock model (Ridge, alpha=10.0 on stationary log-returns/changes of SP500, VIX, and BI Board of Governors announcements).
- **Stronger Volatility Magnitude:** By decreasing the residual Ridge regularization to **alpha=10.0** and increasing the VIX/Spread multipliers (**vix_factor=1.10** and **spread_factor=1.06**), the model is more confident in its fluctuation magnitude, tracking actual volatility spikes closely and reducing OOS RMSE to **269.37**.
