# LAPORAN PRAKTIKUM MATA KULIAH ANALISIS RUNTUN WAKTU
## ANALISIS DAN PERAMALAN KURS USDIDR MENGGUNAKAN TWO-STAGE DECOUPLED RIDGE MODEL DENGAN AKSESERASI GATING MAKRO DAN KOREKSI BIAS DINAMIS

---

## ABSTRAK
Peramalan nilai tukar mata uang, khususnya USDIDR, merupakan tantangan besar dalam ekonometrika keuangan karena sifat datanya yang non-stasioner, sarat akan derau (*noise*), serta rentan terhadap guncangan makro global dan domestik. Laporan praktikum ini membahas pengembangan model peramalan kurs USDIDR harian menggunakan pendekatan **Two-Stage Decoupled Ridge Model**. Model ini memisahkan prediksi tren jangka panjang berbasis lag autoregresif (Stage 1) dari fluktuasi kejutan harian berbasis variabel eksogen makroekonomi (Stage 2). Untuk meningkatkan akurasi, diterapkan sistem gerbang akselerasi (*dynamic gating*) asimetris berbasis VIX dan selisih suku bunga (Spread BI-US), serta korektor bias berbasis rezim volatilitas. Seluruh hyperparameter dioptimalkan menggunakan metode **3-Fold Walk-Forward Time-Series Cross-Validation** pada data training untuk menghindari kebocoran data (*data leakage*). Hasil eksperimen akhir menunjukkan model ini berhasil memprediksi data uji (*out-of-sample*) dengan **RMSE sebesar 272.56**, menunjukkan peningkatan performa yang signifikan dibandingkan model tren dasar statis (RMSE 292.81).

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Nilai tukar Rupiah terhadap Dollar Amerika Serikat (USDIDR) merupakan salah satu indikator makroekonomi terpenting bagi perekonomian Indonesia. Pergerakan kurs ini memengaruhi berbagai sektor, mulai dari perdagangan internasional, inflasi domestik, hingga beban pembayaran utang luar negeri. Oleh karena itu, kemampuan untuk memprediksi pergerakan USDIDR di masa depan secara akurat menjadi sangat krusial bagi pengambil kebijakan (seperti Bank Indonesia) maupun pelaku pasar keuangan.

Namun secara statistik, kurs USDIDR harian sangat sulit diprediksi secara jangka panjang (*multi-step ahead forecasting*). Model runtun waktu linier standar seperti ARIMA sering kali menghasilkan prediksi yang datar (*flat/mean-damped*) ketika digunakan untuk meramal jauh ke depan karena sifat rekursifnya menyaring seluruh volatilitas harian. Di sisi lain, model kecerdasan buatan non-linier kompleks seperti LSTM sering kali mengalami kegagalan generalisasi (*overfitting*) akibat tingginya derau pada data keuangan harian. Laporan ini menawarkan solusi berupa model statistik terdekopel dua-tahap yang stabil, teratur, dan sensitif terhadap dinamika makroekonomi riil.

### 1.2 Tujuan Praktikum
1. Menerapkan konsep stasioneritas dan analisis korelasi parsial (PACF) dalam pembentukan fitur autoregresif USDIDR.
2. Membangun arsitektur *Two-Stage Decoupled Ridge Model* untuk memisahkan komponen tren jangka panjang dan fluktuasi jangka pendek.
3. Mengintegrasikan variabel makroekonomi (VIX dan Spread Suku Bunga) sebagai akselerator prediksi dinamika nilai tukar.
4. Melakukan validasi model yang bersih dari kebocoran data menggunakan skema *3-Fold Walk-Forward Time-Series Cross-Validation*.

---

## BAB II: LANDASAN TEORI

### 2.1 Stasioneritas dan Log-Return
Sebagian besar data runtun waktu keuangan nominal bersifat non-stasioner, yaitu memiliki rata-rata (*mean*) dan varians yang berubah seiring waktu. Meregresikan data non-stasioner secara langsung dapat menyebabkan masalah regresi palsu (*spurious regression*). Untuk mengatasi hal ini, harga nominal $Y_t$ ditransformasikan menjadi log-return harian $r_t$:
$$r_t = \log\left(\frac{Y_t}{Y_{t-1}}\right)$$
Log-return memiliki sifat stasioneritas yang lebih baik dan secara aproksimasi mewakili persentase perubahan harian dari nilai tukar.

### 2.2 Autoregresif (AR) dan Fungsi Autokorelasi Parsial (PACF)
Model Autoregresif berasumsi bahwa nilai variabel pada waktu $t$ dipengaruhi oleh nilai-nilai masa lalunya sendiri pada lag $t-1, t-2, \dots, t-p$. Untuk menentukan ordo lag $p$ yang signifikan, digunakan grafik **Partial Autocorrelation Function (PACF)**. PACF mengukur korelasi bersih antara $r_t$ dan $r_{t-k}$ setelah menghilangkan pengaruh dari seluruh lag yang lebih pendek ($1$ hingga $k-1$). Lag yang menonjol dan melewati batas signifikansi $\pm 1.96/\sqrt{N}$ dipilih sebagai prediktor tren.

### 2.3 Regresi Ridge (L2 Regularization)
Regresi Ridge meminimalkan fungsi kuadrat terkecil dengan menambahkan penalti berupa jumlah kuadrat dari koefisien model (L2 regularization):
$$\text{Loss} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^m w_j^2$$
Dalam konteks peramalan rekursif jangka panjang, penalti $\alpha$ sangat penting untuk mencegah koefisien autoregresif bernilai terlalu besar, sehingga kurva proyeksi jangka panjang tidak mengalami deviasi ekstrem (*meledak*).

### 2.4 Arsitektur Two-Stage Decoupled Model
Model ini membagi tugas prediksi menjadi dua tahap independen:
1. **Model Tren (Stage 1)**: Memprediksi arah return jangka panjang berdasarkan lag autoregresif USDIDR yang stasioner.
2. **Model Kejutan Residu (Stage 2)**: Menghitung selisih (residu) antara aktual return dan hasil prediksi model tren, kemudian memprediksi residu tersebut menggunakan pergerakan variabel eksogen harian global dan domestik (seperti return indeks S&P500, IHSG, harga minyak mentah, dan BI Rate).

### 2.5 Logika Gating Makroekonomi (VIX dan Interest Rate Spread)
Nilai tukar mata uang negara berkembang sangat dipengaruhi oleh aliran modal global (*capital flow*) yang digerakkan oleh dua faktor utama:
1. **Volatilitas Pasar Global (VIX Index)**: Indeks VIX mengukur ketakutan pelaku pasar global. Saat VIX melonjak, terjadi fenomena *risk-off* di mana investor menarik dana dari negara berkembang untuk mengamankan aset mereka di AS, memicu depresiasi Rupiah.
2. **Selisih Suku Bunga (Interest Rate Spread)**: Selisih antara suku bunga Bank Indonesia (BI Rate) dan Federal Funds Rate AS (US Rate). Jika spread menyempit (misal < 0.8%), insentif bagi investor asing untuk memegang Rupiah menurun, memicu aliran modal keluar (*capital flight*).

---

## BAB III: METODOLOGI & DESAIN EKSPERIMEN

### 3.1 Pemrosesan Data & Rekayasa Fitur
Dataset yang digunakan terdiri dari dua bagian: `data_train.csv` (data historis harian) dan `data_test.csv` (data variabel eksogen untuk evaluasi).
Fitur yang dibangun meliputi:
* Log-return dari komoditas/indeks: `SP500_ret`, `GOLD_ret`, `OIL_ret`, `IHSG_ret`, `VIX_ret`.
* Perubahan suku bunga: `bi_rate_change`.
* Lag kausal: Seluruh return eksogen di-shift sebanyak 1 lag untuk menjamin model bersifat kausal. Khusus untuk kebijakan suku bunga BI, digunakan lag 10 hari (`bi_rate_change_lag10`) untuk mengakomodasi lag transmisi kebijakan moneter.

### 3.2 Desain Cross-Validation
Pencarian hyperparameter dilakukan murni pada data training menggunakan metode **3-Fold Walk-Forward Time-Series Cross-Validation**. Skema ini menjaga urutan kronologis waktu agar tidak terjadi kebocoran informasi dari masa depan:
* **Fold 1**: Latih pada baris 1 s.d 1.221 $\rightarrow$ Validasi pada 754 hari berikutnya.
* **Fold 2**: Latih pada baris 1 s.d 1.975 $\rightarrow$ Validasi pada 754 hari berikutnya.
* **Fold 3**: Latih pada baris 1 s.d 2.729 $\rightarrow$ Validasi pada 752 hari terakhir data training.

```text
Visualisasi Pembagian Lipatan (Fold):
Fold 1: [======= Latih =======][--- Validasi 754 hari ---]
Fold 2: [============== Latih ==============][--- Validasi 754 hari ---]
Fold 3: [===================== Latih =====================][--- Validasi 752 hari ---]
```

### 3.3 Pencarian Ruang Parameter (Grid Search)
Eksperimen dilakukan untuk mencari nilai kombinasi terbaik dari tiga hyperparameter utama:
1. **`vix_factor`** (Pengali akselerasi saat pasar cemas): Pilihan $[1.05, 1.10]$
2. **`spread_factor`** (Pengali akselerasi saat spread sempit): Pilihan $[1.02, 1.06]$
3. **`beta`** (Kekuatan penyusutan korektor bias): Pilihan $[0.0, 0.1, 0.2, 0.25]$

Kombinasi parameter yang menghasilkan rata-rata RMSE terkecil di ketiga fold validasi dipilih sebagai parameter optimal.

---

## BAB IV: HASIL DAN PEMBAHASAN

### 4.1 Hasil Grid Search Cross-Validation
Hasil eksekusi proses pencarian parameter di data training disajikan pada tabel di bawah ini:

| VIX Factor | Spread Factor | Bias Beta ($\beta$) | Rata-rata RMSE Validasi |
| :---: | :---: | :---: | :---: |
| 1.05 | 1.02 | 0.00 | 1058.66 |
| 1.05 | 1.02 | 0.10 | 984.17 |
| 1.05 | 1.02 | 0.20 | 917.58 |
| **1.05** | **1.02** | **0.25** | **887.98 (Optimal)** |
| 1.10 | 1.06 | 0.00 | 1065.94 |
| 1.10 | 1.06 | 0.25 | 905.48 |

*Analisis*: Kombinasi parameter **VIX Factor = 1.05**, **Spread Factor = 1.02**, dan **Beta = 0.25** meminimalkan rata-rata RMSE validasi hingga ke angka **887.98**. Aktivasi korektor bias ($\beta = 0.25$) terbukti secara dramatis menurunkan kesalahan prediksi jangka panjang dibanding tanpa koreksi ($\beta = 0.0$).

### 4.2 Pelatihan Ulang dan Performa Out-of-Sample (OOS)
Setelah mengunci parameter optimal, model dilatih kembali menggunakan **100% data training** untuk memanfaatkan data historis terbaru secara maksimal. Selanjutnya, model dijalankan satu kali untuk meramal 754 hari ke depan pada data test.

Hasil evaluasi akhir menunjukkan:
* **RMSE Model Final (Non-Cheating CV)**: **272.5637**
* **RMSE Model Tren Dasar Statis**: **292.8100**

Hal ini menunjukkan bahwa pemisahan komponen shock makro harian beserta akselerasi gating-nya berhasil **mereduksi tingkat kesalahan prediksi (error) sebesar ~6.9%** secara out-of-sample dibandingkan model tren tunggal yang statis.

### 4.3 Pembahasan Justifikasi Empiris Fitur Gating pada Data Training
Untuk memvalidasi kelayakan teoritis dari threshold pembatas gerbang yang digunakan, dilakukan pengujian statistik deskriptif langsung pada data latih:
1. **Analisis VIX (Threshold = 14.0)**:
   * Level VIX = 14.0 mendekati persentil 25% terendah di data training (nilai minimum 9.14, median 16.96).
   * Pada hari-hari di mana VIX > 14.0, nilai rata-rata return harian pelemahan rupiah adalah **0.503%**, sedangkan saat VIX <= 14.0 nilainya hanya **0.321%** (menunjukkan magnitudo depresiasi harian meningkat sebesar **1.56x** pada kondisi pasar cemas). Hal ini menjustifikasi multiplier akselerasi `vix_factor = 1.05`.
2. **Analisis Spread Suku Bunga (Threshold = 0.8% / 80 bps)**:
   * Batas 0.8% merupakan kejadian ekstrim (*tail-risk*) yang mewakili persentil terkecil data latih.
   * Rata-rata return harian USDIDR saat spread menyempit di bawah 0.8% adalah **0.086%**, melonjak hampir **7 kali lipat** dibandingkan ketika spread berada di tingkat aman di atas 0.8% (rata-rata return hanya 0.013%). Hal ini memvalidasi penggunaan multiplier `spread_factor = 1.02` untuk mengantisipasi potensi pelemahan rupiah akibat pelarian modal.

---

## BAB V: KESIMPULAN & SARAN

### 5.1 Kesimpulan
1. Pemodelan runtun waktu USDIDR harian memerlukan pemisahan sinyal yang tegas. Metode *Two-Stage Decoupled Model* terbukti sukses meredam akumulasi derau sekaligus menjaga kepekaan peramalan terhadap perubahan variabel makroekonomi harian.
2. Pengujian murni menggunakan skema *Cross-Validation* berbasis *Walk-Forward* pada data latih berhasil mengidentifikasi parameter optimal tanpa menyebabkan kebocoran data (*zero test-leakage*).
3. Penerapan parameter terpilih pada peramalan out-of-sample menghasilkan nilai **RMSE sebesar 272.56**, membuktikan keandalan model ini secara praktis dan akademis untuk di-submit di papan skor kompetisi maupun analisis laporan.

### 5.2 Saran
1. Untuk penelitian lebih lanjut, pemodelan gerbang asimetris dapat dikembangkan menjadi fungsi kontinu (seperti fungsi sigmoid atau model regresi logistik) alih-alih pembatas biner (*hard threshold*) agar transisi perubahan koefisien lebih halus.
2. Disarankan untuk mengeksplorasi penambahan variabel likuiditas domestik (seperti cadangan devisa atau neraca perdagangan bulanan) sebagai variabel pengendali kestabilan jangka menengah.
