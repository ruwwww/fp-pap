# 4. Forecast: 1 Juni 2023 → 29 Mei 2026

## 4.1 Setup Forecast

| Parameter | Nilai |
|-----------|-------|
| Model | <!-- nama model terbaik --> |
| Periode | 1 Juni 2023 — 29 Mei 2026 |
| Frekuensi | <!-- D/W/M --> |
| Jumlah langkah | <!-- N --> |
| Strategi | One-step-ahead iteratif |

## 4.2 Grafik: Data Aktual + Hasil Forecast

![Forecast Plot](figures/forecast_*.png)

<!-- Embed plot forecast setelah jalankan scripts/forecast.py -->

## 4.3 Statistik Forecast

| Statistik | Nilai |
|-----------|-------|
| Minimum forecast | <!-- val --> |
| Maksimum forecast | <!-- val --> |
| Rata-rata forecast | <!-- val --> |
| Trend | <!-- Naik / Turun / Stagnan --> |

## 4.4 Interpretasi

<!-- UPDATE setelah melihat plot forecast -->

Hasil forecast menunjukkan bahwa:
- Pada periode <!-- start --> hingga <!-- mid -->, terdapat tren **<!-- -->**
- Terdapat pola **<!-- seasonal/cyclical -->** yang terdeteksi model
- <!-- Observasi lain -->

## 4.5 File Submission

```
data/submissions/forecast.csv
```

| Kolom | Deskripsi |
|-------|-----------|
| `date` | Tanggal forecast |
| `forecast` | Nilai prediksi |
