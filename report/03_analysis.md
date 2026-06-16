# 3. Analisis Hasil

## 3.1 Perbandingan Antar Skenario

<!-- UPDATE setelah mendapatkan hasil -->

Tabel di bawah meringkas trend performa model seiring berkurangnya proporsi
data training (dari 80% ke 60%):

| Skenario | Trend RMSE | Observasi |
|----------|-----------|-----------|
| 80/20 → 70/30 | <!-- Naik/Turun --> | <!-- ... --> |
| 70/30 → 60/40 | <!-- Naik/Turun --> | <!-- ... --> |

**Kesimpulan sementara**: <!-- Isi setelah melihat hasil -->

---

## 3.2 Perbandingan ML vs DL

<!-- UPDATE -->

Secara umum, model **<!-- ML/DL -->** menunjukkan performa lebih baik pada dataset ini
karena <!-- alasan: ukuran data, fitur tabular, dll -->.

| Aspek | ML (Model A) | DL (Model B) |
|-------|-------------|-------------|
| RMSE rata-rata | <!-- val --> | <!-- val --> |
| Waktu training | <!-- val --> | <!-- val --> |
| Stabilitas antar skenario | <!-- --> | <!-- --> |
| Interpretabilitas | Tinggi (feature importance) | Rendah (black box) |

---

## 3.3 Efektivitas Ensemble

<!-- UPDATE -->

Ensemble **<!-- naik/turun -->** dibandingkan model terbaik tunggal:

| Model | RMSE (80/20) | vs Model A | vs Model B |
|-------|-------------|-----------|-----------|
| Model A (<!-- nama -->) | <!-- val --> | baseline | - |
| Model B (<!-- nama -->) | <!-- val --> | - | baseline |
| Ensemble C+D | <!-- val --> | <!-- +/-X% --> | <!-- +/-X% --> |

---

## 3.4 Feature Importance

### Top Features (Model A — XGBoost)

![Feature Importance](figures/feature_importance.png)

<!-- Isi setelah generate plot feature importance -->

Fitur-fitur yang paling berpengaruh:
1. `lag_1` — nilai target 1 langkah sebelumnya
2. `rolling_mean_7` — rata-rata bergerak 7 hari
3. <!-- fitur lain -->

---

## 3.5 Analisis Residual

<!-- UPDATE -->

Residual dari model terbaik menunjukkan:
- **Distribusi**: <!-- Normal / Skewed -->
- **Autokorelasi**: <!-- Ada / Tidak ada — cek ACF residual -->
- **Pola sistematis**: <!-- Ada / Tidak ada -->

---

## 3.6 Diskusi

<!-- Isi dengan analisis mendalam -->

### Kelebihan Pendekatan

- ...
- ...

### Keterbatasan

- ...
- ...

### Potensi Peningkatan

- Tambahan fitur eksternal (holiday, ekonomi, cuaca)
- Longer hyperparameter search (lebih banyak trials)
- Stacking multi-level dengan lebih banyak base model
- Transformer-based architecture untuk data yang panjang
