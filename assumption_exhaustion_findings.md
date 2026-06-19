# Assumption Exhaustion Findings

## Scope

Dokumen ini mencatat temuan dari eksperimen asumsi terakhir sebelum melanjutkan.

Fokusnya:

- regime split linear berbasis volatilitas,
- threshold mean-reversion,
- VIX sebagai gate regime.

## Findings

### 1. Regime split linear tidak membantu

Eksperimen `regime_split_linear.py` menunjukkan bahwa memisahkan model linear berdasarkan volatilitas historis tidak memberi lift.

Hasil utama:

- `single_trend_ridge`: `327.29`
- `regime_split_ridge`: `1435.89`

Kesimpulan:

- volatility clustering ada,
- tetapi pemisahan koefisien low-vol/high-vol dengan skema ini tidak stabil,
- jadi regime split linear bukan jalur yang produktif saat ini.

### 2. Threshold mean-reversion membantu

Eksperimen `threshold_intervention_experiment.py` menguji mean-reversion yang hanya aktif di ekstrem.

Hasil utama:

- `baseline_trend_ridge`: `327.29`
- `threshold_mean_reversion_1.50`: `314.86`
- validation baseline RMSE: `411.30`
- validation best RMSE: `322.07`

Kesimpulan:

- mean-reversion memang lebih kuat ketika diasumsikan asimetris,
- threshold `1.50` paling baik dari grid yang diuji,
- ini mendukung ide bahwa intervensi/mean-reversion USDIDR tidak bekerja mulus setiap hari.

### 3. VIX lebih cocok sebagai gate daripada prediktor

Benchmark tambahan menunjukkan model terbaik muncul saat VIX dipakai untuk memilih antara dua model:

- model tren terbaik: `ar_plus_trend`
- model threshold ekstrem: `threshold_mean_reversion_1.50`

Hasil benchmark OOS:

- `286.55` RMSE
- gate terbaik: `VIX_lag1 > 27.5` pilih model threshold, selain itu pilih model tren

## Important Caveat

Angka `286.55` adalah **benchmark-only** dan **bukan final deployable result**.

Alasannya:

- threshold gate dipilih dengan melihat OOS actual,
- jadi ada leakage evaluasi pada tahap pemilihan gate,
- angka ini valid sebagai upper bound diagnosis, bukan estimasi generalisasi yang bersih.

## Interpretation

Temuan ini menyiratkan bahwa:

- asumsi mean-reversion memang belum habis dieksploitasi,
- bentuk yang lebih tepat adalah threshold/non-linear trigger, bukan koefisien linear global,
- VIX berfungsi lebih baik sebagai saklar regime daripada sebagai feature kontinu.

## Next Step

Langkah berikutnya yang benar adalah:

1. pilih gate threshold dari validation internal train,
2. fit final model pada full train,
3. evaluasi sekali di OOS test,
4. baru bandingkan dengan `286.55` sebagai benchmark kasar.
