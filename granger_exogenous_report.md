# Granger Exogenous Investigation

## Setup

- Data: `data_train.csv`
- Target: `USDIDR.diff()`
- Exogenous: `GOLD`, `SP500`, `OIL`, `VIX`, `US_rate`, `BI_rate`, `IHSG`, `CPI`
- Test: bivariate Granger causality on differenced series up to lag 90

## Forward Direction

Hasil utama: beberapa market variables punya predictive signal yang sangat kuat.

| feature | best lag | best p-value | note |
| --- | ---: | ---: | --- |
| `SP500` | 6 | `7.0e-55` | sangat kuat |
| `VIX` | 20 | `3.7e-44` | sangat kuat |
| `IHSG` | 4 | `2.3e-35` | sangat kuat |
| `US_rate` | 79 | `4.7e-15` | kuat, tapi jauh lag-nya |
| `GOLD` | 4 | `6.9e-14` | kuat |
| `OIL` | 2 | `2.5e-05` | masih signifikan |
| `BI_rate` | 35 | `0.1268` | tidak signifikan |
| `CPI` | 2 | `0.2582` | tidak signifikan |

Semua market variables utama (`SP500`, `VIX`, `IHSG`, `GOLD`) signifikan di banyak lag, bukan cuma satu lag.

## Reverse Direction

Saya cek juga arah balik `USDIDR.diff() -> exog.diff()`.

| feature | best p exog -> USDIDR | best p USDIDR -> exog |
| --- | ---: | ---: |
| `SP500` | `7.0e-55` | `7.0e-03` |
| `VIX` | `3.7e-44` | `2.7e-03` |
| `IHSG` | `2.3e-35` | `3.0e-04` |
| `US_rate` | `4.7e-15` | `1.0e-06` |
| `GOLD` | `6.9e-14` | `1.5e-04` |
| `OIL` | `2.5e-05` | `1.1e-01` |
| `BI_rate` | `1.3e-01` | `5.7e-03` |
| `CPI` | `2.6e-01` | `1.7e-02` |

Interpretasi:

- ini bukan bukti kausalitas murni,
- banyak variabel market kemungkinan menangkap common stress / global shock,
- tapi sebagai predictor, variabel-variabel itu jelas tidak kosong.

## What This Means

1. `SP500`, `VIX`, `IHSG`, `GOLD`, `OIL`, dan `US_rate` layak dipertimbangkan sebagai exogenous inputs.
2. `BI_rate` dan `CPI` tidak menunjukkan signal Granger yang kuat di setup ini.
3. Untuk SSM, jangan pakai same-day exog saja.
   - pakai lag spesifik,
   - dan kalau bisa coefficient yang time-varying / regime-dependent.

## Recommended Lag Set

Start point yang paling masuk akal:

- `SP500_lag6`
- `VIX_lag20`
- `IHSG_lag4`
- `GOLD_lag4`
- `OIL_lag2`
- `US_rate_lag79`

## Modeling Implication

SSM yang paling relevan untuk investigasi berikutnya adalah:

```text
y_t = trend_t + beta_t * X_{t-lag} + eps_t
beta_t = beta_{t-1} + noise
```

Bukan vanilla ARIMAX dengan koefisien tetap.

## Holdout Sanity Check

Saya juga coba ridge regression time-split di `data_train.csv` untuk cek apakah lag exog terpilih memberi lift praktis.

| model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| `AR_only` | `72.40` | `53.12` | `0.36%` |
| `AR_plus_selected_exog` | `74.60` | `56.16` | `0.38%` |
| `AR_plus_all_exog` | `75.90` | `57.02` | `0.39%` |

Interpretasi:

- sinyal Granger ada,
- tapi model linear sederhana belum memanfaatkan sinyal itu lebih baik dari AR-only,
- jadi efeknya kemungkinan butuh interaction/regime gating/timing yang lebih pintar.
