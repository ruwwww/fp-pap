# 🤖 AI Agent Guide — EAS Time Series Project (v2)

> **Baca dokumen ini sebelum mengubah apapun.**
> Semua definisi model dan hyperparameter ada di **satu config terpusat**.

---

## 📐 Architecture Overview

```
eas/
├── config/
│   ├── config.yaml            ← Global: target_col, date_col, paths, forecast
│   └── models_config.yaml     ← SEMUA model + params + search_space di sini
│
├── scripts/                   ← Pure Python, background-friendly
│   ├── train.py               ← Training (parallel ML support)
│   ├── search.py              ← Optuna hyperparameter search
│   ├── forecast.py            ← Future forecasting
│   ├── evaluate.py            ← Metrics + report generation
│   ├── runs.py                ← Command-line interface to view/manage runs
│   └── run_all.py             ← Full pipeline orchestrator
│
├── src/                       ← Core library
│   ├── artifacts/             ← NEW: Immutable run artifact storage system
│   │   ├── artifact_store.py  ← Creates & manages timestamped run folders
│   │   └── run_record.py      ← Log model, configs, metrics, plots (read-only)
│   ├── data/                  ← loader.py, preprocessor.py
│   ├── features/              ← feature_engineering.py
│   ├── models/                ← base_model.py, ml_models.py, dl_models.py, ensemble_models.py
│   ├── evaluation/            ← metrics.py
│   └── visualization/         ← plots.py
│
├── runs/                      ← NEW: Immutable Run Folders
│   ├── <run_id>/              ← Timestamped runs
│   │   ├── config_snapshot.yaml
│   │   ├── models_config_snapshot.yaml
│   │   ├── run_metadata.json
│   │   ├── model.joblib       ← Saved serialized model
│   │   ├── metrics.json
│   │   ├── params.json
│   │   ├── predictions.csv
│   │   ├── run_report.md      ← Auto-generated Markdown documentation for this run
│   │   └── plots/
│   └── index.md               ← Automatically rebuilt markdown index of all runs
│
├── report/                    ← Markdown laporan
│   ├── 00_cover.md
│   ├── 01_methodology.md      ← Isi manual
│   ├── 02_results.md          ← AUTO-GENERATED oleh evaluate.py
│   ├── 03_analysis.md         ← Isi manual setelah lihat hasil
│   ├── 04_forecast.md         ← Isi manual setelah forecast
│   ├── figures/               ← Embed gambar di sini
│   └── tables/                ← CSV per scenario
│
├── results/
│   ├── plots/                 ← Semua plot PNG (headless/Agg)
│   ├── metrics/               ← results_summary.csv
│   └── search/                ← best_params.yaml + .db (Optuna)
│
└── data/
    ├── raw/                   ← CSV dari Kaggle → taruh di sini
    └── submissions/           ← forecast.csv output
```

---

## 🎯 Centralized Config — Cara Kerja

### `config/config.yaml` — Global Settings
```yaml
project:
  target_column: "value"     # ← UPDATE dengan nama kolom target
  date_column: "date"        # ← UPDATE dengan nama kolom tanggal
forecast:
  start_date: "2023-06-01"
  end_date: "2026-05-29"
  frequency: "D"             # D=daily, W=weekly, M=monthly
```

### `config/models_config.yaml` — Semua Model + Search Space
Setiap model memiliki 3 bagian:
```yaml
xgboost:
  enabled: true              # ← toggle on/off tanpa ubah kode
  class: "src.models.ml_models.XGBoostModel"
  category: "ML"
  parallel_safe: true        # ← bisa dijalankan di subprocess
  params:                    # ← hyperparameter default
    n_estimators: 300
    learning_rate: 0.05
  search_space:              # ← search space untuk Optuna
    n_estimators:
      type: int
      low: 100
      high: 800
    learning_rate:
      type: float
      low: 0.005
      high: 0.3
      log: true              # ← log-scale sampling
```

---

## ⚡ Quick Commands

```bash
# Setup
pip install -r requirements.txt
pip install -e .

# Full pipeline (search → train → evaluate → forecast)
python scripts/run_all.py --data data/raw/train.csv

# Full pipeline, parallel ML, 100 trials search
python scripts/run_all.py --data data/raw/train.csv --parallel --workers 4 --trials 100

# Background execution (Windows)
Start-Process python -ArgumentList "scripts/run_all.py --data data/raw/train.csv" -RedirectStandardOutput logs/run.log -NoNewWindow

# Quick smoke (ML only, 80/20 only, 10 trials)
python scripts/run_all.py --data data/raw/train.csv --skip-category DL --scenarios 80_20 --trials 10 --skip-forecast

# Search only
python scripts/search.py --models xgboost,lightgbm --trials 100 --jobs 4

# Train only with best params
python scripts/train.py --data data/raw/train.csv --use-best-params --parallel --workers 4

# Train only specific models
python scripts/train.py --data data/raw/train.csv --models xgboost,lstm

# Forecast only
python scripts/forecast.py --model xgboost --use-best-params

# Generate report
python scripts/evaluate.py
```

---

## ⚠️ Anti-Leakage Rules

| Rule | Benar | Salah |
|------|-------|-------|
| Rolling stats | `series.shift(1).rolling(w).mean()` | `series.rolling(w).mean()` |
| Lag features | `series.shift(n)` dimana n ≥ 1 | `series.shift(0)` |
| Scaler fit | `scaler.fit(X_train)` saja | `scaler.fit(df)` |
| Train/Test split | Temporal (n baris pertama) | Random shuffle |
| Forecast iteratif | Lag dari prediksi sebelumnya | Lag dari data masa depan |

**`TimeSeriesFeatureEngineer` sudah menerapkan ini. Jangan bypass.**

---

## 🧩 Cara Menambah Model Baru

### 1. Tambah kelas di `src/models/ml_models.py` (atau `dl_models.py`):
```python
class ElasticNetModel(BaseTimeSeriesModel):
    def __init__(self, params=None):
        super().__init__(name="ElasticNet", category="ML", params=params or {})
    def fit(self, X_train, y_train):
        from sklearn.linear_model import ElasticNet
        self.model = ElasticNet(**self.params)
        self.model.fit(X_train, y_train)
        return self
    def predict(self, X_test):
        return self.model.predict(X_test)
    def get_params(self):
        return self.params
```

### 2. Daftarkan di `config/models_config.yaml`:
```yaml
ml_models:
  elastic_net:
    enabled: true
    class: "src.models.ml_models.ElasticNetModel"
    category: "ML"
    parallel_safe: true
    params:
      alpha: 1.0
      l1_ratio: 0.5
    search_space:
      alpha:
        type: float
        low: 0.001
        high: 100.0
        log: true
      l1_ratio:
        type: float
        low: 0.0
        high: 1.0
```

### 3. Selesai — `train.py` dan `search.py` otomatis pick up model baru.

---

## 📊 Output Files

| File | Dibuat oleh | Keterangan |
|------|-------------|-----------|
| `results/metrics/results_summary.csv` | `train.py` | Semua metrik, semua model |
| `results/search/best_params.yaml` | `search.py` | Best hyperparameter per model |
| `results/search/<model>.db` | `search.py` | Optuna study SQLite |
| `results/plots/*.png` | `train.py` + `evaluate.py` | Semua plot (headless) |
| `report/02_results.md` | `evaluate.py` | Tabel hasil otomatis |
| `report/tables/*.csv` | `evaluate.py` | Per-scenario CSV |
| `data/submissions/forecast.csv` | `forecast.py` | Submission Kaggle |

---

## 🔄 Typical Workflow untuk AI Agent

```
1. Baca config/config.yaml  → cek target_column & date_column
2. Cek data/raw/            → pastikan CSV ada
3. python scripts/search.py --category ML --trials 50   (cari HP dulu)
4. python scripts/train.py --use-best-params --parallel  (train semua)
5. python scripts/evaluate.py                            (buat laporan)
6. python scripts/forecast.py --model <best>             (forecast)
7. Update report/01_methodology.md dengan detail model
8. Update report/03_analysis.md dengan interpretasi hasil
```

---

## 🐛 Debugging

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError` | `pip install -e .` dari root project |
| `KeyError: target_column` | Update `config/config.yaml` |
| LSTM OOM | Kurangi `lookback` atau `batch_size` di `models_config.yaml` |
| Optuna `TrialPruned` | Normal — trial di-skip karena performa buruk |
| Plot tidak muncul | Script pakai `matplotlib.use("Agg")` — headless, cek di `results/plots/` |
| Parquet error | Install `pyarrow`: `pip install pyarrow` |
