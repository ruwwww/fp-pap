# ⏱️ EAS Time Series Forecasting — Project Environment

> **Mata Kuliah**: Pemodelan dan Analisis Prediktif (PAP)
> **Deadline Kaggle**: Senin, 22 Juni 00:00 WIB
> **Deadline Pengumpulan**: Senin, 22 Juni 12:00 WIB

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone / navigate to project
cd "C:\kuliahh maseh\pap\eas"

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install project as local package (enables clean imports)
pip install -e .
```

### 2. Configure Your Dataset

```yaml
# config/config.yaml  ← EDIT THESE:
project:
  target_column: "your_target_column"   # ← column to predict
  date_column: "your_date_column"       # ← datetime column
```

### 3. Put Data in `data/raw/`

```bash
# Option A: Download manually from Kaggle and place CSV here
data/raw/train.csv
data/raw/test.csv

# Option B: Use Kaggle API
pip install kaggle
# Put your kaggle.json at C:\Users\<you>\.kaggle\kaggle.json
python -c "
from src.data.loader import DataLoader
loader = DataLoader()
loader.download_kaggle(competition='<competition-name>')
"
```

### 4. Run EDA First

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 5. Run Full Benchmark (all models, all scenarios)

```bash
# Via notebook:
jupyter notebook notebooks/04_Benchmarking.ipynb

# Or via CLI:
python run_benchmark.py --data data/raw/train.csv --target value --date date
```

### 6. Generate Forecast

```bash
jupyter notebook notebooks/05_Forecasting.ipynb
```

---

## 📋 Model Assignment (sesuai instruksi)

| Model | Type | Default | Sesuai instruksi |
|-------|------|---------|-----------------|
| **Model A** | Machine Learning | XGBoost | Pilih 1 ML model |
| **Model B** | Deep Learning | LSTM | Pilih 1 DL model (LSTM/GRU/CNN/RNN) |
| **Model C** | Ensemble base | XGBoost | Boleh sama dengan A atau B |
| **Model D** | Ensemble base | LSTM | **WAJIB** DL (LSTM/GRU/CNN/RNN) |
| **Ensemble** | C + D combined | VotingEnsemble | Evaluasi sebagai 1 model |

---

## 🎛️ Enabling / Disabling Models

Edit `config/models_config.yaml`:

```yaml
ml_models:
  xgboost:
    enabled: true   # ← set false to skip
  lightgbm:
    enabled: false  # ← set true to include
```

---

## 📊 Output Structure

After benchmarking, you'll find:

```
results/
├── plots/
│   ├── model_A_80_20.png           # Actual vs Predicted plots
│   ├── model_B_80_20.png
│   ├── ensemble_80_20.png
│   ├── heatmap_RMSE.png            # Model × Scenario heatmaps
│   ├── heatmap_MAPE.png
│   └── results_table.png           # Rekapitulasi table
└── metrics/
    └── results_summary.csv         # All metrics, all models, all scenarios
```

---

## 🔍 MLflow Experiment Tracking

```bash
mlflow ui --port 5000
# → Open http://localhost:5000
```

---

## 📁 Project Structure

```
eas/
├── config/                  # YAML configuration
├── src/                     # Core library
│   ├── data/                # Loading & preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # ML, DL, Ensemble models
│   ├── evaluation/          # Metrics (RMSE, MAPE, MAE, R²)
│   └── visualization/       # Plot functions
├── benchmarking/            # BenchmarkRunner + FutureForecaster
├── notebooks/               # Jupyter workflow
├── data/                    # raw/, processed/, submissions/
├── results/                 # plots/, metrics/, models/
├── docs/                    # Documentation
│   └── AI_AGENTS_GUIDE.md  # Guide for AI agents
├── run_benchmark.py         # CLI entrypoint
├── requirements.txt
└── setup.py
```

---

## ⚠️ Anti-Leakage Checklist

Before submitting, verify:

- [ ] All rolling features use `shift(1)` before `.rolling()`
- [ ] All lag features use `shift(n)` where `n ≥ 1`
- [ ] Scalers fit ONLY on training data (`preprocessor.fit_transform(train)`)
- [ ] Train/Test split is **temporal** (first N rows = train, not random)
- [ ] No future data used in any normalization or aggregation

---

## 🛠️ Common Commands

```bash
# Quick ML-only benchmark (skip DL for speed)
python run_benchmark.py --data data/raw/train.csv --target value --date date --skip lstm,gru,cnn_lstm

# Test single model
python run_benchmark.py --data data/raw/train.csv --target value --date date --only xgboost

# No MLflow
python run_benchmark.py --data data/raw/train.csv --no-mlflow
```

---

## 📚 Documentation

- **[AI Agent Guide](docs/AI_AGENTS_GUIDE.md)** — Architecture, rules, task examples for AI agents
- **[models_config.yaml](config/models_config.yaml)** — All model parameters
- **[config.yaml](config/config.yaml)** — Project settings

---

*Generated for EAS PAP — Time Series Forecasting Project*
