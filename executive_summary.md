# Executive Summary

- Target terbaik: `log_return`
- Regime shift: ya, kuat
- Validation terbaik untuk seleksi model: expanding
- Strategi deployment: rolling retraining
- Baseline terbaik: Naive (RMSE 73.3689)
- LightGBM terbaik: target `log_return` (mean RMSE 243.7348)
- Apakah LightGBM mengalahkan Naive? belum tentu pada holdout ini; hasil tergantung target dan validation fold

## Rekomendasi Final
- Gunakan LightGBM pada log return sebagai target utama.
- Pakai rolling retraining, bukan sekali fit lalu dipakai lama.
- Simpan Naive sebagai sanity baseline; jika LightGBM tidak konsisten mengalahkan Naive, model level tidak layak diprioritaskan.
- Fokus pada fitur lag/momentum/rolling dan spread makro, bukan tuning agresif dulu.