# USD/IDR Forecasting Framework

## Problem Statement
Forecasting USD/IDR harian ini bukan sekadar masalah model, tetapi masalah struktur sinyal.

- Target level USD/IDR bersifat trend-heavy dan non-stationary.
- Lag USD/IDR sangat dominan.
- Exogenous macro signal ada, tetapi lemah, delay, dan tidak stabil lintas rezim.

## Operational Requirements
Framework ini harus data-driven, bukan asumsi bentuk model.

- Decomposition harus dibuktikan dari data.
- Residual harus diuji apakah benar predictable.
- Regime harus diidentifikasi secara statistik.
- Exogenous effect harus diperlakukan sebagai interaction, bukan sekadar additive effect.
- Lag structure harus dipelajari, bukan diasumsikan.

## Main Diagnosis
Model global seperti LightGBM cenderung flatten karena:

- mereka menangkap autoregressive structure target lebih dulu,
- exogenous tidak memberi lift konsisten pada semua periode,
- relationship macro berubah antar rezim,
- level forecasting memaksa model mengekstrapolasi tren jangka panjang.

## Reframed Formulation
Jangan prediksi USD/IDR level secara langsung.

Gunakan dekomposisi yang dipelajari dari data:

```text
USDIDR_t = Trend_t + Residual_t
```

atau dalam bentuk return:

```text
log(USDIDR_t) - log(USDIDR_{t-1}) = AR_component_t + Macro_residual_t
```

Interpretasi awal:

- `Trend_t` menangkap drift dan AR structure.
- `Residual_t` adalah bagian kecil yang mungkin dipengaruhi macro.

Namun separability harus diuji, bukan diasumsikan.

### Decomposition Checks

- rolling drift stability
- structural break segmentation
- state-space or adaptive trend estimation

Jika trend tidak stabil, jangan pakai decomposition statik.
Trend harus menjadi learned component.

## Recommended Modeling Stack

### 1. Trend / AR Model
Forecast komponen utama dulu dengan history USD/IDR.

- baseline: naive / linear AR
- tujuan: menangkap dominant path

### 2. Residual Model
Hitung residual dari trend model, lalu modelkan residual dengan exogenous.

- fitur: OIL, GOLD, SP500, IHSG, VIX, CPI, BI_RATE, US_RATE
- gunakan lag 1-5 hari dan rolling features
- fokus pada error correction, bukan level prediction

### Residual Diagnostics

Sebelum residual dimodelkan, cek dulu apakah residual benar-benar informative:

- autocorrelation residual
- mutual information dengan exogenous
- stability across regimes

Jika residual mendekati noise, maka macro modeling tidak akan berhasil karena signal memang tidak ada.

### 3. Regime Gate
Aktifkan macro hanya saat regime teridentifikasi secara statistik.

- learned clustering seperti KMeans atau HMM
- volatility threshold yang diturunkan dari distribusi data
- structural break detection seperti CUSUM atau Bai-Perron

Jangan pakai rule manual. Regime harus deterministic atau learnable.

### 4. Reconstruction
Gabungkan kembali forecast trend + residual:

```text
forecast = trend_forecast + residual_forecast
```

## Interaction Modeling
Exogenous effect harus dimodelkan sebagai interaksi, bukan additive saja.

Residual sebaiknya dipelajari sebagai:

```text
Residual = f(macro x regime_state x lag_structure)
```

Artinya:

- efek macro bisa berubah tergantung regime
- efek macro bisa berubah tergantung slope trend
- efek macro bisa muncul hanya pada lag tertentu

## Why This Should Work Better

- Exogenous menjadi korektor, bukan penentu utama.
- Model tidak dipaksa mengekstrapolasi level mentah.
- Relationship yang berubah-ubah bisa dipisah per rezim.
- AR dominance ditangani eksplisit.
- Lag discovery dan interaction effects dipelajari dari data.

## Evaluation Layer

Jangan evaluasi hanya dengan RMSE.

- conditional mutual information gain over AR baseline
- SHAP stability across time
- feature importance variance across folds

Ini penting karena kadang RMSE tidak berubah banyak, tetapi signal sebenarnya muncul hanya pada slice rezim tertentu.

## Failure Modes to Avoid

- signal absence: macro memang tidak berguna
- signal misalignment: lag salah
- signal instability: regime shift
- model incapacity: extrapolation failure
- direct level forecasting with one global model
- same-day exogenous assumption tanpa cek lag
- over-reliance on boosting to discover unstable macro relations
- tuning sebelum diagnosis sinyal

## Failure Classification

Kalau model gagal, klasifikasikan dulu ke salah satu ini:

- signal absence
- signal misalignment
- signal instability
- model incapacity

## Final Answer
Transformasi utama bukan hanya trend-residual decomposition, tetapi regime-conditioned interaction modeling with learned lag structure, where exogenous variables are treated as conditional correction signals whose effect is only activated when statistical dependency is proven stable within specific regimes.

Itu cara paling masuk akal agar exogenous variables berubah dari noise menjadi sinyal yang conditional dan interpretable.
