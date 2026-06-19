# Exogenous Deep Dive

## Reassessment
Kesimpulan awal bahwa USD/IDR hanya AR-dominant perlu dibuat lebih hati-hati.

Yang benar dari data:

- exogenous tidak cukup kuat untuk memenangkan forecast **level** secara konsisten,
- tetapi exogenous **punya signal arah** yang lebih jelas daripada signal magnitude,
- dan signal itu muncul lebih kuat pada sebagian regime dan volatility slice.

## What Improved the Most

### Directional predictability
Model sign arah perubahan USD/IDR naik cukup jelas ketika exogenous ditambahkan di atas lag target.

Walk-forward AUC untuk arah berubah dari sekitar:

- `0.53-0.62` untuk AR-only

menjadi:

- `0.65-0.74` untuk AR + exogenous

contoh:

- 2016: `0.5787 -> 0.6569`
- 2018: `0.5707 -> 0.6989`
- 2022: `0.5851 -> 0.7379`
- 2023: `0.5347 -> 0.6747`

## What Did Not Improve Much

### Magnitude / RMSE
Exogenous masih tidak konsisten memperbaiki error level.

Ini konsisten dengan:

- effect size yang kecil,
- signal yang conditional,
- dan hubungan yang berubah antar periode.

## Where Exogenous Helps Most

### 1. High-volatility slices
Conditional AUC cenderung lebih tinggi pada volatility quartile yang lebih tinggi.

Contoh pooled test slice:

- vol quartile 0: `0.5401 -> 0.7238`
- vol quartile 1: `0.5572 -> 0.6830`
- vol quartile 2: `0.6432 -> 0.7529`

Ini menunjukkan macro lebih berguna saat market stress / re-pricing.

### 2. Trend-slope slices
Ketika slope USD/IDR berbeda antar periode, exogenous juga memberi lift pada arah.

Contoh:

- slope quartile 0: `0.5337 -> 0.6219`
- slope quartile 1: `0.6896 -> 0.7657`
- slope quartile 2: `0.5708 -> 0.6894`
- slope quartile 3: `0.6284 -> 0.7389`

## Variable-Level Signal

Signal paling konsisten masih datang dari risk-sensitive variables:

- `SP500`
- `VIX`
- `IHSG`

Temuan penting:

- `VIX` cenderung positif dan stabil pada rolling correlation,
- `SP500` dan `IHSG` cenderung negatif,
- rates seperti `US_rate` dan `BI_rate` jauh lebih lemah dan lebih tidak stabil.

## Interpretation

Exogenous di sini bukan driver utama level USD/IDR, tapi juga bukan noise murni.

Lebih tepatnya:

- exogenous memberikan **directional correction signal**,
- efeknya **conditional**, bukan global,
- dan paling relevan saat volatilitas atau re-pricing tinggi.

## Revised Conclusion

Problem ini bukan sepenuhnya macro-informative, tapi juga bukan macro-useless.

Kesimpulan yang lebih akurat:

> USD/IDR daily forecasting is AR-dominant for magnitude, but exogenous variables contain conditional directional signal that becomes materially useful in high-volatility and regime-sensitive slices.

Artinya, jika tujuan hanya RMSE level, baseline AR/persistence tetap sangat sulit dikalahkan.
Jika tujuan mengeluarkan signal ekonomi yang benar-benar ada, maka problem harus diformulasikan sebagai **conditional direction / regime-aware correction**, bukan pure level forecasting.
