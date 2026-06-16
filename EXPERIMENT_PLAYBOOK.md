# 🎯 INSTRUKSI EXPERIMENT AGENT (PLAYBOOK)
## Proyek: Peramalan USD/IDR (Rupiah Resilience)

Dokumen ini berisi daftar perintah eksperimen terstruktur yang harus dieksekusi secara otomatis oleh **AI Experiment Agent** untuk menghasilkan model, data prediksi, grafik, dan tabel metrik yang sesuai dengan persyaratan laporan terlampir.

---

## 🏃 ALUR PERINTAH EKSPERIMEN (SEQUENTIAL COMMANDS)

### Tahap 1: Hyperparameter Optimization (Optuna Search)
Cari parameter terbaik untuk semua model agar bebas dari bias manual.
```bash
# 1. Cari parameter terbaik untuk Machine Learning (Model A / Model C)
python scripts/search.py --category ML --trials 50 --jobs 4

# 2. Cari parameter terbaik untuk Deep Learning (Model B / Model D)
python scripts/search.py --category DL --trials 20
```

### Tahap 2: Benchmarking & Training (3 Skenario × Semua Model)
Latih Model A, Model B, Model C, dan Model D (serta Voting/Stacking Ensemble) di bawah skenario temporal split:
- **Skenario 1**: 80% Train - 20% Test
- **Skenario 2**: 70% Train - 30% Test
- **Skenario 3**: 60% Train - 40% Test

```bash
# Train semua model menggunakan parameter terbaik dari hasil Optuna secara paralel
python scripts/train.py --data data/raw/data_train.csv --use-best-params --parallel --workers 4
```

### Tahap 3: Evaluasi & Sinkronisasi Laporan Otomatis
Kumpulkan data metrik (RMSE, MAPE, MAE, R²) dari seluruh skenario dan model untuk dimasukkan ke laporan.
```bash
# Generate tabel rekapitulasi, grafik evaluasi, dan perbarui file laporan
python scripts/evaluate.py
```

### Tahap 4: Future Forecasting (Prediksi Masa Depan Kaggle)
Prediksi nilai USD/IDR mulai 1 Juni 2023 hingga 29 Mei 2026 berdasarkan dataset test (`data/raw/data_test.csv`).
```bash
# Lakukan forecasting menggunakan model terbaik (misal: ensemble)
python scripts/forecast.py --model ensemble --use-best-params
```

---

## 📂 PEMETAAN HASIL EKSPERIMEN KE LAPORAN (`report/`)

Setiap output run disimpan ke folder `runs/` secara immutable, dan ringkasannya akan dipetakan langsung ke struktur laporan di folder [report/](file:///C:/kuliahh%20maseh/pap/eas/report):

| Komponen Bab Laporan | Sumber File Hasil Eksperimen | Keterangan & Bentuk Visualisasi |
|---|---|---|
| **BAB III - 3.2 Dataset** | `data/raw/data_train.csv` | Tampilkan snapshot data mentah dan karakteristik deskriptifnya. |
| **BAB III - 3.6 Prosedur Validasi** | [config.yaml](file:///C:/kuliahh%20maseh/pap/eas/config/config.yaml) | Definisi pembagian skenario train-test split (80/20, 70/30, 60/40). |
| **BAB IV - 4.1 s/d 4.3 (Hasil Model)** | `runs/<run_id>/run_report.md` | Memuat parameter terbaik yang digunakan & grafik evaluasi masing-masing model. |
| **BAB IV - 4.4 Tabel Perbandingan** | [02_results.md](file:///C:/kuliahh%20maseh/pap/eas/report/02_results.md) | Tabel rekapitulasi RMSE, MAPE, MAE, R² lintas 3 skenario yang di-update oleh `evaluate.py`. |
| **BAB IV - 4.6 Hasil Forecast** | `results/plots/forecast_*.png` | Plot hasil peramalan data masa depan (Juni 2023 - Mei 2026). |
| **Lampiran (Submission)** | `data/submissions/forecast.csv` | File csv keluaran akhir untuk diunggah ke Scoreboard Kaggle. |

---

## 🔍 MONITORING EXPERIMENT RUNS (CLI tools)
Gunakan perintah berikut untuk melacak status eksperimen yang sedang berjalan di background:
```bash
# Tampilkan daftar seluruh run yang sukses disimpan
python scripts/runs.py list

# Tampilkan metrik dan parameter detail run spesifik
python scripts/runs.py show <RUN_ID>
```
