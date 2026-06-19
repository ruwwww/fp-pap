# USD/IDR Forecasting Architecture

## Principle
Only keep a component if the data shows it helps.

This means:

- no regime layer unless it improves walk-forward results,
- no state-space model unless it beats simpler trend handling,
- no exogenous feature unless it adds stable lift over autoregressive baseline.

## What The Data Already Says

- USD/IDR is strongly autoregressive.
- Exogenous variables are useful mainly for direction, not for large RMSE reduction.
- `SP500`, `VIX`, and `IHSG` carry the clearest macro signal.
- `CPI`, `BI_rate`, and `US_rate` are stepwise and update only on certain dates.
- Exogenous signal is strongest in stress periods and around change dates.

## Practical Architecture

### 1. Baseline model
Predict USD/IDR change from its own history first.

Use simple inputs:

- `USDIDR` lags
- rolling mean
- rolling std
- recent momentum

This is the reference model. If nothing beats it consistently, do not add complexity.

### 2. Exogenous branch
Add macro variables only after they are transformed into usable signals.

Split them into two groups:

- `market variables`: `OIL`, `GOLD`, `SP500`, `IHSG`, `VIX`
- `state variables`: `CPI`, `BI_rate`, `US_rate`

For market variables:

- use daily changes,
- use lagged changes,
- use spike flags if the move is unusually large.

For state variables:

- use the level,
- use change flags,
- use days since last change,
- use event-day flags around the update date.

### 3. Lag search
Do not assume one common lag.

Search each exogenous variable separately and keep only lags that are stable across folds.

Rules:

- if a lag works only once, discard it,
- if a feature only helps in one regime, treat it as conditional,
- if same-day input is weaker than lagged input, use the lagged version.

### 4. Direction model
Build a separate model for up/down movement.

This is important because the data shows exogenous variables help direction more than magnitude.

### 5. Event correction model
For days when `BI_rate`, `CPI`, or `US_rate` changes, add a small correction model.

This is not a regime system.
It is just an event-day adjustment for known update dates.

### 6. Final forecast
Combine the outputs as:

```text
final forecast = autoregressive baseline + exogenous correction
```

The correction should be small unless the model is confident.

## Where State-Space Fits

State-space is optional.

Use it only if it shows clear evidence of helping to separate trend from noise.

Accept it only if:

- it improves walk-forward RMSE over a simple AR baseline,
- residual autocorrelation drops,
- the improvement holds across several folds,
- it does not collapse on the 2020 and 2022-2023 shift periods.

If those checks fail, do not use state-space.

## What To Avoid

- one global model with all features treated equally,
- same-day macro as if every feature had identical timing,
- hard regime labels without evidence,
- complex decomposition without out-of-sample gain,
- adding exogenous variables just because they are economically sensible.

## Recommended Final Design

The most honest design is:

1. autoregressive baseline for magnitude,
2. exogenous branch for direction and event-day correction,
3. optional state-space only if it proves useful.

That keeps the system simple, testable, and tied to the actual behavior of the data.
