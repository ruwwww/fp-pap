# USD/IDR Signal Diagnosis

## Bottom Line
USD/IDR daily forecasting di dataset ini **AR-dominant**, bukan macro-informative.

Exogenous variables memang mengandung sinyal, tetapi sinyalnya **lemah, conditional, dan tidak cukup stabil** untuk mengalahkan autoregressive baseline secara konsisten.

Kesimpulan jujur:

- problem ini **tidak signal-free**,
- tetapi **signal macro terlalu kecil dan terlalu tidak stabil** untuk menjadi source of durable predictive lift,
- sehingga performa terbaik secara alami akan tetap **dekat ke autoregressive baseline**.

## What the Data Says

### 1. Target sendiri sangat kuat sebagai predictor

Forecast level USD/IDR dari `lag-1` saja sudah sangat kompetitif.

Walk-forward check menunjukkan bahwa menambah exogenous ke AR1 biasanya tidak membantu:

| period | AR1 RMSE | AR1 + exo same-day | AR1 + exo lag1 | AR1 + exo diff |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 73.811 | 79.593 | 78.668 | 75.069 |
| 2020 | 175.424 | 180.735 | 179.513 | 176.672 |
| 2021 | 73.921 | 82.332 | 82.470 | 74.877 |
| 2022 | 60.248 | 61.880 | 60.978 | 64.155 |
| 2023 | 69.643 | 68.826 | 69.677 | 70.249 |

Result: exogenous variants are **not consistently better** than AR1.

### 2. Daily autoregressive signal exists, but is limited

For first difference target:

- lag-1 autocorr: `-0.273`
- AR(1..10) in-sample R2 only reaches about `0.092`

So there is structure, but it is not rich enough to explain everything.

### 3. Exogenous signal is real, but weak and unstable

Best lag search on the full sample shows the strongest deployable relationships are mostly short-lag and modest:

- `SP500` best corr with target diff: about `-0.221` at lag 1
- `VIX` best corr with target diff: about `0.192` at lag 1
- `IHSG` best corr with target diff: about `-0.141` at lag 1
- `GOLD` best corr with target diff: about `-0.080` at lag 1
- `US_rate` best corr is weaker and often unstable

Rolling 252D correlation summary:

- `SP500`: mean `-0.183`, sign positive only `~5%` of windows
- `VIX`: mean `0.186`, sign positive `~92%` of windows
- `IHSG`: mean `-0.139`, sign positive `~12%` of windows
- `GOLD`: mean `-0.102`, sign positive `~14%` of windows
- `US_rate`: mean `-0.022`, sign positive `~47%` of windows
- `BI_rate`: mean `-0.019`, sign positive `~40%` of windows

Interpretation:

- equity risk proxies have directionally plausible signal,
- rate variables are much weaker,
- effect sizes are modest,
- stability is not strong enough to create durable model lift.

### 4. Residual signal after AR is tiny

Using AR5 on `USDIDR diff`:

- AR5 in-sample R2: `0.0863`

Mutual information between exogenous and AR residual is very small:

- strongest: `VIX`
- then `GOLD`, `SP500`
- most others are near zero

That means residual is **not strongly macro-explainable**.

### 5. Regime instability is real

The target level shifts materially across periods, and the mean level rises strongly from early sample to later sample.

This means a global learned mapping has poor transferability.

## Failure Classification

The failure mode is not one thing.

- **signal absence**: some macro features are simply too weak
- **signal misalignment**: lag differs by variable and by period
- **signal instability**: relationships change across regimes
- **model incapacity**: global boosting struggles with extrapolation and regime transfer

## Final Conclusion

USD/IDR daily forecasting here is **fundamentally AR-dominant and signal-limited**.

There is some exogenous signal, especially from risk-sensitive proxies like `SP500` and `VIX`, but it is **not strong or stable enough** to dominate the autoregressive baseline. The best achievable daily forecast will therefore remain close to persistence/AR structure, with macro likely usable only as a **small conditional correction** in selected regimes.

In short:

> exogenous variables are informative in parts of the sample, but not enough to make USD/IDR daily forecasting strongly macro-driven.
