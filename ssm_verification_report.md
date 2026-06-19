# SSM Verification Report

## Question
Apakah USD/IDR data ini benar-benar mendukung pendekatan hidden-state / state-space yang lebih baik daripada baseline autoregressive?

## Verified Findings

### 1. Hidden regimes are statistically present
Markov switching on `USDIDR diff` strongly supports more than one regime.

BIC results:

- 2 regimes: `39575.75`
- 3 regimes: `39220.13`
- 4 regimes: `39181.24`

Conclusion: regime-switching structure is real, and 4 states fit best among the tested options.

### 2. The detected states are economically meaningful
The 4-state model separates the series into volatility tiers.

State summary on `USDIDR diff`:

- state 0: mean abs move `10.20`
- state 1: mean abs move `39.04`
- state 2: mean abs move `55.69`
- state 3: mean abs move `148.93`

The highest-vol state concentrates heavily in known stress periods:

- 2020 share in highest-vol state: `0.6947`
- 2022 share in highest-vol state: `0.0385`
- 2023 share in highest-vol state: `0.0000`

This means the model finds one crisis-like state, but not all later periods map to that same state.

### 3. State-space style forecasting did not beat AR baseline
Out-of-sample level forecast comparison:

- Naive RMSE: `470.965`
- AR5 RMSE: `72.765`
- Local level RMSE: `476.147`
- Local linear trend RMSE: `889.545`
- SARIMAX(1,1,0) + exogenous RMSE: `473.533`

Conclusion: the tested state-space baselines did **not** improve forecasting performance over AR5.

### 4. Time-varying regression also failed to add lift
Previously tested dynamic regression on `USDIDR diff` with exogenous inputs produced worse OOS RMSE than fixed regression and AR baselines.

Conclusion: time-varying coefficients are statistically plausible, but not yet practically beneficial in this setup.

## Final Verification

The data does support **hidden regime structure**.

But the data does **not** support the claim that a basic state-space / Kalman-style forecast, as tested here, will automatically beat a simple autoregressive baseline.

So the verified answer is:

> hidden states exist, but vanilla SSM formulations are not sufficient; the forecast edge still comes from AR structure, not from the state-space layer alone.
