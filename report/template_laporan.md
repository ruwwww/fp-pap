# LAPORAN PENELITIAN PEMODELAN DAN ANALITIKA PREDIKTIF
## Peramalan Nilai Tukar Rupiah (USD/IDR) Menggunakan Model Machine Learning, Deep Learning, dan Ensemble Learning

**Kelompok**: 99  
**Kelas**: PAP1 D  
**Anggota Kelompok**:  
1. Nama Lengkap (NRP)  
2. Nama Lengkap (NRP)  
3. Nama Lengkap (NRP)  

**Mata Kuliah**: Pemodelan dan Analitika Prediktif  
**Dosen Pengampu**: Prof. Dr. Wiwik Anggraeni, S.Si., M.Kom.  

**DEPARTEMEN SISTEM INFORMASI**  
**FAKULTAS TEKNOLOGI ELEKTRO DAN INFORMATIKA CERDAS**  
**INSTITUT TEKNOLOGI SEPULUH NOPEMBER**  
**GENAP 2025/2026**  

---

## DAFTAR ISI
- [BAB I. PENDAHULUAN](#bab-i-pendahuluan)
- [BAB II. TINJAUAN PUSTAKA](#bab-ii-tinjauan-pustaka)
- [BAB III. METODOLOGI PENELITIAN](#bab-iii-metodologilog-penelitian)
- [BAB IV. HASIL DAN PEMBAHASAN](#bab-iv-hasil-dan-pembahasan)
- [BAB V. KESIMPULAN DAN SARAN](#bab-v-kesimpulan-dan-saran)
- [DAFTAR PUSTAKA](#daftar-pustaka)

---

## BAB I. PENDAHULUAN

### 1.1 Latar Belakang
Nilai tukar Rupiah terhadap Dolar Amerika Serikat (USD/IDR) merupakan salah satu indikator ekonomi yang sangat penting bagi pemerintah, pelaku bisnis, investor, maupun masyarakat umum. Pergerakan kurs dipengaruhi oleh berbagai faktor global dan domestik, seperti harga komoditas dunia, kondisi pasar saham, tingkat inflasi, suku bunga, serta sentimen risiko pasar keuangan. Projek ini bertujuan memodelkan time series univariate/multivariate untuk memprediksi pergerakan USD/IDR menggunakan model Machine Learning (Model A), Deep Learning (Model B), dan Ensemble Learning (Model C dan D).

### 1.2 Rumusan Masalah
1. Bagaimana performa model Machine Learning (Model A: XGBoost/LightGBM) dibandingkan dengan model Deep Learning (Model B: LSTM/GRU) untuk memprediksi USD/IDR?
2. Bagaimana performa model Ensemble Learning (Model C + D) jika dibandingkan dengan model tunggal terbaik?
3. Bagaimana stabilitas dan sensitivitas performa model di bawah 3 skenario train-test split (80/20, 70/30, 60/40)?

### 1.3 Tujuan
1. Mengetahui performa model Machine Learning (Model A) dibandingkan dengan model Deep Learning (Model B).
2. Mengevaluasi efektivitas penggabungan model ML & DL via Ensemble Learning.
3. Menganalisis pengaruh persentase split data training terhadap akurasi peramalan.

### 1.4 Manfaat
- **Bagi Mahasiswa**: Memberikan pengalaman praktis dalam penerapan model peramalan time series (ML, DL, Ensemble) pada data keuangan riil tanpa kebocoran data (anti-leakage).
- **Bagi Institut**: Menambah portofolio riset dan implementasi analitika prediktif di Institut Teknologi Sepuluh Nopember.
- **Bagi Masyarakat/Praktisi**: Menyediakan rekomendasi pemodelan berbasis data historis untuk mitigasi risiko pergerakan nilai tukar USD/IDR.

---

## BAB II. TINJAUAN PUSTAKA

### 2.1 Time Series Forecasting: Konsep dan Karakteristik
Deret waktu (time series) adalah serangkaian pengamatan yang dicatat secara berurutan dalam interval waktu tertentu. Karakteristik utama data deret waktu meliputi tren (kecenderungan jangka panjang), variasi musiman (fluktuasi periodik), siklus, dan fluktuasi tak beraturan.

### 2.2 Model Machine Learning untuk Time Series (Model A)
Model Machine Learning seperti XGBoost dan LightGBM menggunakan pendekatan berbasis pohon keputusan yang ditingkatkan secara gradual (gradient boosting). Data deret waktu dimodelkan secara autoregressive dengan merumuskan data lag, statistik bergerak (rolling statistics), dan data kalender sebagai fitur tabular input.

### 2.3 Model Deep Learning (Model B & D)
Jaringan syaraf tiruan rekuren (RNN) seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU) memiliki gerbang sel khusus (gates) untuk menyimpan informasi jangka panjang. Model DL sangat andal dalam menangkap dependensi temporal non-linier dalam sekuensial deret waktu yang panjang.

### 2.4 Ensemble Learning
Ensemble learning memadukan beberapa model basis (base models) untuk menghasilkan prediksi akhir yang lebih stabil dan akurat. Pada perancangan ini, prediksi dari Model C (ML) dan Model D (Deep Learning) akan digabungkan secara voting/stacking/rata-rata tertimbang untuk menyeimbangkan kelebihan masing-masing arsitektur.

---

## BAB III. METODOLOGI PENELITIAN

### 3.1 Alur Penelitian
Pengumpulan Data -> Prapemrosesan & Feature Engineering (Anti-Leakage) -> Split Skenario (80/20, 70/30, 60/40) -> Tuning Hyperparameter (Optuna) -> Training & Benchmarking -> Evaluasi Metrik -> Peramalan Masa Depan.

### 3.2 Gambaran Umum Dataset
Dataset historis harian (Business Days) periode Januari 2010 hingga Mei 2026. Data training memiliki 8 exogenous features:
- OIL (Minyak Mentah)
- GOLD (Emas)
- SP500 & IHSG (Indeks Saham)
- VIX (Volatility Index)
- CPI (Consumer Price Index)
- BI_rate & US_rate (Suku bunga)

### 3.3 Eksplorasi dan Analisis Data
Pengecekan statistik deskriptif dasar, visualisasi tren pergerakan target `USDIDR` dan korelasi antar fitur eksogen.

### 3.4 Prapemrosesan Data
- **3.4.1 Pembersihan**: Menangani anomali nilai dan memastikan sequential harian kerja (Business Days).
- **3.4.2 Feature Engineering**: Pembuatan lag target, rolling mean/std dengan melakukan `shift(1)` terlebih dahulu untuk menghindari **data leakage**.
- **3.4.3 Transformasi**: Normalisasi fitur menggunakan `MinMaxScaler` yang hanya di-fit pada data training.

### 3.5 Perancangan Model
- **3.5.1 Model A (Machine Learning)**: Model berbasis tree dengan tuning parameters via Optuna.
- **3.5.2 Model B (Deep Learning)**: Arsitektur sekuensial LSTM/GRU dengan input tensor 3D `[samples, lookback, features]`.
- **3.5.3 Ensemble Learning (Model C + D)**: Integrasi prediksi dari model ML (C) dan DL (D).

### 3.6 Prosedur Validasi dan Evaluasi
Evaluasi kinerja model di bawah 3 skenario temporal split menggunakan metrik: RMSE, MAPE, MAE, dan R².

### 3.7 Implementasi dan Tools
Implementasi menggunakan Python (Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, Optuna) dengan output file model `.joblib` / `.h5`.

---

## BAB IV. HASIL DAN PEMBAHASAN

*(Bab ini otomatis disinkronisasi dengan performa run dari sistem artifact. Jalankan `python scripts/evaluate.py` untuk memuat tabel dan plot evaluasi secara dinamis)*

---

## BAB V. KESIMPULAN DAN SARAN

### 5.1 Kesimpulan
Memberikan intisari performa model terbaik dari 3 skenario yang dijalankan serta model mana yang paling andal untuk memprediksi pergerakan USD/IDR.

### 5.2 Saran
Rekomendasi pengembangan lebih lanjut seperti penambahan fitur sentimen berita ekonomi atau penambahan model transformer.

---

## DAFTAR PUSTAKA
- Shumway, R. H., & Stoffer, D. S. (2019). *Time Series Analysis and Its Applications*. Springer.
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: principles and practice*. OTexts.
