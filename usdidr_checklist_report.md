# USD/IDR Checklist Report

## A. Data Quality

- Jumlah data train: 3498
- Rentang tanggal train: 2010-01-04 s/d 2023-05-31
- Duplicate tanggal: 0
- Missing value: 0
- Frekuensi tanggal dominan: 1 hari dan 3 hari (weekend gap)

### Statistik USDIDR Train

- Min: 8292.70
- Max: 16504.80
- Mean: 12504.65
- Median: 13335.00
- Std: 2228.66

## B. EDA

- Trend train kuat: slope/day = 2.0435, R2 = 0.8575
- Trend test actual juga naik: slope/day = 2.4093, R2 = 0.7845
- STL trend strength: 0.9989
- STL seasonal strength: 0.0385
- Distribusi level tidak normal, skew negatif ringan

## C. Train-Test Shift

- Mean train: 12504.65
- Mean test: 16108.58
- Std train: 2228.66
- Std test: 611.33
- Max test: 17840.00 > max train: 16504.80
- Kesimpulan: distribution shift ada

## F. Stationarity

- Level: tidak stationary (ADF p=0.715823, KPSS p=0.01)
- First difference: stationary
- Log return: stationary

## G. Autocorrelation

- Level sangat autocorrelated
- Log return punya autocorrelation awal negatif di lag 1
- Lag penting pada return relatif pendek

## H. Volatility

- Rolling std 30D: mean 0.006081
- Rolling std 90D: mean 0.006227
- Ada volatility clustering

## I. Baseline Forecast

Validation split: last 20% train

- Naive: RMSE 73.369, MAE 53.484, MAPE 0.366%
- Linear Regression: RMSE 1816.756, MAE 1780.846, MAPE 12.198%
- ARIMA(1,1,2): RMSE 470.070, MAE 405.222, MAPE 2.769%

Best baseline: Naive

## L. Final Diagnosis

| Komponen | Hasil |
| --- | --- |
| Trend | Kuat |
| Seasonality | Lemah |
| Stationary Level | Tidak |
| Stationary Return | Ya |
| Distribution Shift | Ya |
| Volatility Clustering | Ya |
| Residual Pattern | Ada |
| Best Baseline | Naive |
| Best ML | Belum diuji |

## Rekomendasi

- Forecast level langsung kurang cocok untuk jangka panjang.
- Fokus ke return/delta atau hybrid trend + residual model.
- Karena test actual jelas lebih tinggi, rolling retraining atau regime-aware approach layak diprioritaskan.
