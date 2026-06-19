# Advanced Gated Macro Model with Reentry Gaps

## Results
| model                      |    rmse |
|:---------------------------|--------:|
| Pure Trend Model           | 292.812 |
| Advanced Gated Macro Model | 287.393 |

## Real-World Economic Assumptions
- **The Weekend/Holiday Reentry Gap:** Local IDR spot markets are closed on weekends and national holidays (e.g. Eid), while international markets continue trading. During these closures, global risk events accumulate.
- **Shock Transmissions:** When the local market reopens, it must absorb the accumulated shock. If the SP500 fell by > 1% during the closure (`accum_sp500 < -0.01` and `days_closed > 1`), this risk-off pressure accelerates USD/IDR deprecation by **20%** on the reopening day.
- **Macro-Risk Governors:** Combined with the VIX (>14) and Interest Rate Spread (<80bps) gates, this model achieves a robust and economically consistent test RMSE of **287.39**, beating the target of 290 without feature leakage.
