# 1. Metodologi

## 1.1 Dataset

| Atribut | Keterangan |
|---------|-----------|
| **Nama** | Rupiah Resilience: USD/IDR Forecasting |
| **Sumber** | Kaggle — [link kompetisi](https://www.kaggle.com/t/e5266e2645fa4e56984ee34cab1ee81c) |
| **Target** | `USDIDR` — Nilai tukar USD terhadap IDR |
| **Frekuensi** | Harian (hari kerja / business days, tidak termasuk weekend & libur) |
| **Periode Train** | 4 Januari 2010 — 31 Mei 2023 (3.498 baris) |
| **Periode Test Kaggle** | 1 Juni 2023 — 29 Mei 2026 (778 baris) |
| **Jumlah Fitur** | 8 exogenous + 1 target |

### Variabel Dataset

| Kolom | Deskripsi | Satuan |
|-------|-----------|--------|
| `Date` | Tanggal hari kerja | — |
| `OIL` | Harga minyak mentah dunia | USD/barrel |
| `GOLD` | Harga emas dunia | USD/troy oz |
| `USDIDR` | **Target** — Nilai tukar USD/IDR | IDR |
| `SP500` | Indeks saham S&P 500 (USA) | poin |
| `IHSG` | Indeks Harga Saham Gabungan Indonesia | poin |
| `VIX` | Volatility Index (fear index) | — |
| `CPI` | Consumer Price Index Indonesia | — |
| `BI_rate` | Suku bunga Bank Indonesia | % |
| `US_rate` | Suku bunga Federal Reserve AS | % |

### Statistik Deskriptif (data_train.csv)

| Statistik | OIL | GOLD | USDIDR | SP500 | IHSG | VIX | CPI | BI_rate | US_rate |
|-----------|-----|------|--------|-------|------|-----|-----|---------|---------|
| **Count** | 3498 | 3498 | 3498 | 3498 | 3498 | 3498 | 3498 | 3498 | 3498 |
| **Mean** | 71.40 | 1464.11 | 12494.24 | 2452.26 | 5195.94 | 18.71 | 4.23 | 5.64 | 0.76 |
| **Std** | 22.41 | 259.30 | 2248.77 | 1014.55 | 1130.44 | 7.18 | 1.54 | 1.32 | 1.10 |
| **Min** | -37.63 | 1050.80 | 888.11 | 1022.58 | 2475.48 | 9.14 | 1.56 | 3.50 | 0.05 |
| **Max** | 123.70 | 2051.50 | 16504.80 | 4796.56 | 7318.02 | 82.69 | 6.41 | 7.75 | 5.06 |

> **Catatan**: Tidak ada missing values pada data training.
> OIL pernah bernilai negatif (-37.63) pada April 2020 (COVID-19 crash).

### Visualisasi Data

![Time Series Overview](../results/plots/01_timeseries_overview.png)

---

## 1.2 Preprocessing

### 1.2.1 Penanganan Missing Values

Data training tidak memiliki missing values. Namun karena dataset hanya berisi
hari kerja (business days), terdapat "gap" akhir pekan dan hari libur yang
tidak memengaruhi model karena data sudah sequential.

Untuk lag features: nilai NaN yang muncul pada baris-baris awal (akibat `shift`)
dihapus menggunakan `.dropna()` sebelum training.

### 1.2.2 Deteksi Outlier

OIL bernilai negatif (-37.63) pada 20 April 2020 — ini adalah kejadian nyata
(WTI oil futures crash) dan **dipertahankan** sebagai informasi historis valid.

### 1.2.3 Scaler

Untuk model ML (XGBoost, LightGBM, dll): tidak menggunakan scaler karena
tree-based model tidak memerlukan normalisasi.

Untuk model DL (LSTM, GRU): MinMaxScaler diaplikasikan di dalam kelas model
(`BaseDLModel._scale()`), di-fit hanya pada training data.

---

## 1.3 Feature Engineering

> **Anti-Leakage**: Seluruh fitur dihitung menggunakan `shift(1)` sebelum
> operasi rolling/EMA sehingga tidak menggunakan informasi masa depan.

### Fitur yang Dibuat

| Kelompok | Fitur | Keterangan |
|----------|-------|-----------|
| **Target Lags** | `usdidr_lag_{1,2,3,5,10,20,30,60}` | Nilai USDIDR t-k hari lalu |
| **Rolling USDIDR** | `usdidr_rmean_{5,10,20,60}`, `rmean_std`, `rmin`, `rmax` | Statistik bergerak (shift+1) |
| **EMA USDIDR** | `usdidr_ema_{5,20,60}` | Exponential moving average |
| **Diff USDIDR** | `usdidr_diff_{1,5,20}` | Selisih nilai (momentum) |
| **Exog Lags** | `{oil,gold,sp500,...}_lag_{1,5}` | Lag variabel eksogen |
| **Exog Rolling** | `{col}_rmean_{5,20}`, `{col}_diff_1` | Rolling & diff variabel eksogen |
| **Ratio Features** | `gold_oil_ratio`, `sp500_vix_ratio`, `ihsg_sp500_ratio`, `rate_spread` | Interaksi antar variabel |
| **Calendar** | `day_of_week`, `month`, `quarter`, `is_month_end`, dll. | Fitur temporal |
| **Cyclical** | `month_sin/cos`, `dow_sin/cos` | Encoding siklus |

**Catatan domain penting**:
- `rate_spread = BI_rate - US_rate`: diferensial suku bunga → indikator kurs
- `gold_oil_ratio`: rasio komoditas → proxy risk sentiment
- `sp500_vix_ratio`: risk appetite pasar AS
- `usdidr_vs_ema20`: deviasi dari rata-rata bergerak → mean-reversion signal

---

## 1.4 Pembagian Data (Train-Test Split)

**Internal evaluation** (dari `data_train.csv` saja, untuk mengukur performa model):

| Skenario | Train | Test | Periode Train Aktual | Periode Test Aktual |
|----------|-------|------|---------------------|-------------------|
| Skenario 1 | 80% (2799 baris) | 20% (699 baris) | Jan 2010 — ~Aug 2019 | ~Aug 2019 — Mei 2023 |
| Skenario 2 | 70% (2449 baris) | 30% (1049 baris) | Jan 2010 — ~Jun 2017 | ~Jun 2017 — Mei 2023 |
| Skenario 3 | 60% (2099 baris) | 40% (1399 baris) | Jan 2010 — ~Jan 2015 | ~Jan 2015 — Mei 2023 |

**Kaggle submission**: model ditraining pada **seluruh** `data_train.csv`,
prediksi dilakukan pada `data_test.csv` (1 Jun 2023 — 29 Mei 2026).

---

## 1.5 Model yang Digunakan

### Model A — Machine Learning

**Model**: <!-- XGBoost / LightGBM / RandomForest / dll → tentukan setelah benchmark -->

**Alasan pemilihan**: <!-- isi setelah benchmark -->

**Hyperparameter terpilih** (setelah Optuna search):

| Parameter | Nilai Default | Nilai Terbaik |
|-----------|--------------|--------------|
| `n_estimators` | 300 | *diisi setelah search* |
| `learning_rate` | 0.05 | *diisi setelah search* |
| `max_depth` | 6 | *diisi setelah search* |
| `subsample` | 0.8 | *diisi setelah search* |

### Model B — Deep Learning

**Model**: <!-- LSTM / GRU / CNN-LSTM → tentukan setelah benchmark -->

**Arsitektur**:
```
Input (lookback=30, features=N)
  → LSTM(128, return_sequences=True) + Dropout(0.2)
  → LSTM(64)
  → Dense(32, relu)
  → Dense(1)
```

**Hyperparameter terpilih**:

| Parameter | Nilai Default | Nilai Terbaik |
|-----------|--------------|--------------|
| `lookback` | 30 | *diisi setelah search* |
| `units` | [128, 64] | *diisi setelah search* |
| `dropout` | 0.2 | *diisi setelah search* |
| `learning_rate` | 0.001 | *diisi setelah search* |
| `batch_size` | 32 | *diisi setelah search* |

### Ensemble Learning — Model C + D

**Strategi**: <!-- Voting / Stacking / Blending → tentukan setelah benchmark -->

| Peran | Model | Kategori |
|-------|-------|---------|
| **Model C** | <!-- nama → boleh sama dengan A atau B --> | ML atau DL |
| **Model D** | <!-- nama → WAJIB DL: LSTM/GRU/CNN/RNN --> | **Deep Learning** |

---

## 1.6 Metrik Evaluasi

| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Error dalam satuan IDR; sensitivitas terhadap outlier |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_i - \hat{y}_i}{y_i}\|$ | Error dalam %; mudah diinterpretasi bisnis |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Error rata-rata dalam IDR; robust terhadap outlier |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proporsi variansi yang dijelaskan; mendekati 1 = baik |

**Konteks**: Untuk USD/IDR sekitar ~14.000–16.000:
- RMSE < 200 IDR → sangat baik
- MAPE < 1% → sangat baik
- MAPE < 2% → baik

---

## 1.7 Hyperparameter Tuning (Optuna)

```bash
# Search XGBoost dan LightGBM, 100 trials, 4 jobs paralel
python scripts/search.py --models xgboost,lightgbm --trials 100 --jobs 4

# Search LSTM, 30 trials
python scripts/search.py --models lstm --trials 30
```

- **Algoritma**: Tree-structured Parzen Estimator (TPE)
- **Metrik optimasi**: RMSE (minimisasi)
- **Validation**: Hold-out 20% terakhir dari training split
- **Storage**: SQLite di `results/search/<model>.db` (bisa di-resume)

---

## 1.8 Forecast (Kaggle Submission)

Berbeda dengan forecast iteratif biasa:
- `data_test.csv` **sudah berisi** semua variabel eksogen (OIL, GOLD, SP500, dll.)
  untuk periode 1 Jun 2023 — 29 Mei 2026
- Model **tidak perlu** memprediksi variabel eksogen masa depan
- Strategi: train pada seluruh `data_train.csv`, predict langsung pada `data_test.csv`
- Untuk lag features: data train digunakan sebagai konteks (prepended)

```bash
python scripts/forecast.py --model xgboost --use-best-params
```
