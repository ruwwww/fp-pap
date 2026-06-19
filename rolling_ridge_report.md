# Adaptive Walk-Forward Rolling Ridge (Walking Beta) with Bounded Exogenous Ranks

## Results
| model                        |     rmse |
|:-----------------------------|---------:|
| Static Pure Trend Model      |  346.487 |
| Adaptive Rolling Ridge Model | 1584.97  |

## Real-World Economic Assumptions
- **Eliminating OOD Shift via Percentile Ranks:** By replacing absolute commodity/stock prices (e.g. SP500, OIL, GOLD) with a rolling 252-day percentile rank (`_rank_252`), we restrict the feature space strictly between 0 and 1, completely preventing wild linear extrapolations.
- **Shock Magnitudes:** Normalizing daily returns by 20-day rolling standard deviations (`_shock`) contextually measures market panic across regimes.
- **Walking Beta (Adaptive Linear Window):** Fitting the Ridge model walk-forward on a rolling window of 504 days (2 years) allows coefficients to dynamically adjust to changing market conditions (such as monetary policy shifts), making the model highly adaptive without overfitting.
