# Train Set vs Test Set Structural Comparison Report

## 1. Target Level Distribution (USDIDR)
- **Train Period (2010 - May 2023):** Range: [8292.70, 16504.80] | Mean: 12504.65 | Std: 2228.66
- **Test Period (June 2023 - May 2026):** Range: [14833.00, 17840.00] | Mean: 16108.58 | Std: 611.33
- **OOD Level Indicator:** The minimum price in the test set (14833.00) is close to the average price of the train set, and the maximum (17840.00) is far above anything seen in training.

## 2. Daily Log-Return Dynamics (USDIDR Returns)
- **Train Period Daily Mean Return:** 0.000134 (Annualized: 3.39%)
- **Test Period Daily Mean Return:** 0.000225 (Annualized: 5.66%)
- **Train Period Daily Volatility (Std):** 0.006795
- **Test Period Daily Volatility (Std):** 0.006677
- **Insight:** The daily depreciation rate of USDIDR in the test set is significantly higher than the train set, but the daily volatility is actually slightly lower. This indicates a **persistent structural drift upwards** rather than high-frequency noise spikes.

## 3. Exogenous Market Regimes
- **S&P 500 Mean:** Train: 2452.26 | Test: 5706.38 (Massive positive shift)
- **VIX Mean (Global Panic):** Train: 18.80 | Test: 17.10 (Low volatility regime on average)

## 4. Key Takeaways & Core Quant Insight
- **The Illusion of Chaos:** While we assume the test period is more chaotic (high VIX), statistically, VIX in the test set has a lower average than the historical training set. The test set is characterized by a stable but persistent bull market in US equities (SP500 averaging 4500+) combined with a steady depreciation of the Rupiah.
- **Why ARIMA / LSTM Fails:** Simple statistical models expect mean reversion when USDIDR goes to new highs because they are anchored to the historical training mean. In the test set, the target level drifts constantly upward without returning to the mean.
- **The Essence of Exploration:** To predict USD/IDR successfully, the model must not rely on historical mean reversion of levels. Instead, it must map a persistent directional drift rate, which is why our log-return formulation combined with risk gates (accelerators) succeeds in staying on track.
