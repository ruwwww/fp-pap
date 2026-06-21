# BAB II. TINJAUAN PUSTAKA

## 2.1 Konsep Peramalan Runtun Waktu Keuangan dan Stasioneritas
Peramalan runtun waktu (*time series forecasting*) pada instrumen keuangan memiliki karakteristik yang sangat berbeda dibandingkan data fisik atau iklim. Data harga nominal seperti kurs mata uang asing ($Y_t$) umumnya bersifat non-stasioner, yaitu memiliki nilai rata-rata (*mean*) dan varians yang berubah seiring waktu (Shumway & Stoffer, 2019). Penggunaan data non-stasioner dalam analisis regresi dapat menyebabkan masalah *spurious regression* (regresi palsu) di mana koefisien determinasi ($R^2$) terlihat tinggi namun tidak memiliki arti struktural secara empiris.

Untuk mencapai stasioneritas, data harga nominal ditransformasikan menjadi log-return harian ($r_t$):
$$r_t = \log\left(\frac{Y_t}{Y_{t-1}}\right)$$
Log-return mewakili persentase perubahan harga harian secara kontinu dan memiliki distribusi yang lebih mendekati distribusi stasioner (rata-rata konstan di sekitar nol dan varians stabil), sehingga valid untuk dimodelkan menggunakan metode linier maupun non-linier.

## 2.2 Model Machine Learning Tradisional: Ridge Regression (Regulasi L2)
Model autoregresif (AR) memproyeksikan nilai masa depan berdasarkan kombinasi linier dari nilai-nilai historisnya sendiri. Namun, pada peramalan rekursif jangka panjang (*multi-step ahead recursive forecasting*), estimasi parameter menggunakan Ordinary Least Squares (OLS) rentan terhadap *overfitting* akibat adanya multikolinieritas antar lag. 

Regresi Ridge mengatasi kelemahan ini dengan menambahkan penalti kuadrat dari koefisien (regulasi L2) ke dalam fungsi kerugian (*loss function*):
$$\text{Loss}_{\text{Ridge}} = \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \|\mathbf{w}\|_2^2$$
Di mana $\alpha$ adalah hyperparameter regulasi. Regulasi L2 memaksa koefisien autoregresif ($w_j$) menyusut mendekati nol secara halus. Hal ini sangat krusial dalam peramalan rekursif karena mencegah akumulasi kesalahan prediksi jangka pendek membesar secara eksponensial di akhir horizon waktu (Hastie et al., 2009).

## 2.3 Deep Learning untuk Runtun Waktu: Gated Recurrent Unit (GRU)
Gated Recurrent Unit (GRU) adalah varian dari Recurrent Neural Network (RNN) yang dirancang untuk mengatasi masalah *vanishing gradient* pada data runtun waktu yang panjang (Cho et al., 2014). GRU menyederhanakan arsitektur Long Short-Term Memory (LSTM) dengan menggabungkan *cell state* dan *hidden state* menjadi satu variabel, serta menggunakan dua gerbang utama:
* **Reset Gate ($r_t$)**: Menentukan seberapa banyak informasi masa lalu yang harus dilupakan.
* **Update Gate ($z_t$)**: Menentukan seberapa banyak informasi dari *hidden state* sebelumnya yang akan diteruskan ke *hidden state* baru.

Matematika di balik pembaruan sel GRU adalah sebagai berikut:
$$z_t = \sigma(W_z x_t + U_z h_{t-1})$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1})$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
Keunggulan GRU terletak pada kemampuannya menangkap dependensi non-linier jangka pendek pada residu (kejutan pasar) dengan parameter yang lebih sedikit dibandingkan LSTM, menjadikannya efisien untuk data harian.

## 2.4 Konsep Pemodelan Terdekopel Dua-Tahap (Two-Stage Decoupled Modeling)
Perilaku pergerakan nilai tukar emerging markets terdiri atas komponen tren jangka panjang yang persisten dan fluktuasi jangka pendek yang acak akibat guncangan eksternal (*high-frequency noise*). Jika kedua sinyal ini digabungkan secara langsung pada satu model tunggal, model akan kesulitan membedakan antara tren dan derau.

Pendekatan *Two-Stage Decoupled Modeling* memecah proses ini:
1. **Stage 1 (Trend Model)**: Memodelkan tren dasar log-return nilai tukar hanya menggunakan lag-lag autoregresif terpilih yang memiliki autokorelasi bersih signifikan berdasarkan Partial Autocorrelation Function (PACF).
2. **Stage 2 (Residual Shock Model)**: Mengisolasi varians sisa (residu) dari model tren yang tidak dapat dijelaskan oleh pergerakan historis harga, lalu memodelkannya menggunakan kejutan harian dari variabel eksogen makroekonomi eksternal.

## 2.5 Landasan Ekonomi Makro: Kepekaan Asimetris Rupiah terhadap VIX dan Suku Bunga
Nilai tukar mata uang emerging markets seperti Rupiah sangat rentan terhadap arah arus modal internasional (*global capital flows*). Hubungan ini dipengaruhi oleh dua indikator makroekonomi utama:
1. **Global Volatility (VIX Index) dan Sentimen Risk-Off**:
   Indeks VIX mengukur ekspektasi volatilitas pasar saham S&P500 dan sering disebut sebagai "indeks ketakutan global". Ketika VIX meningkat (di atas level ambang batas historis ~14.0), sentimen investor global beralih menjadi *risk-off* (menolak risiko). Investor asing akan beramai-ramai melikuidasi portofolio mereka di negara berkembang dan memindahkan asetnya ke safe-haven (seperti US Dollar atau Obligasi Pemerintah AS). Fenomena ini mempercepat depresiasi Rupiah secara asimetris (Krugman, 1979).
2. **Monetary Interest Rate Differential (BI Rate - US Fed Funds Rate Spread)**:
   Selisih suku bunga (*Spread*) bertindak sebagai kompensasi risiko (*risk premium*) bagi investor asing yang memegang aset Rupiah. Berdasarkan teori *Uncovered Interest Rate Parity* (UIP), menyempitnya spread bunga domestik terhadap bunga penopang global (US Fed Rate) di bawah batas kritis psikologis (~0.8% atau 80 bps) akan menghilangkan insentif investor untuk menahan Rupiah. Akibatnya terjadi pelarian modal keluar (*capital flight*), meningkatkan permintaan terhadap US Dollar secara masif dan menekan Rupiah melemah tajam.

## 2.6 Mekanisme Koreksi Bias Berbasis Rezim Volatilitas
Peramalan jangka panjang secara rekursif langkah-demi-langkah ke depan (*iterative recursive forecasting*) secara inheren akan mengakumulasikan kesalahan prediksi (*forecast drift*). Hal ini dikarenakan setiap prediksi langkah ke $t+1$ menggunakan nilai prediksi hari $t$ sebagai input lag-nya.

Untuk menstabilkan lintasan peramalan jangka panjang, diterapkan modul **Bias Correction** dinamis. Akumulasi bias peramalan dimodelkan berdasarkan umur peramalan (*forecast age*), kemiringan kurva (*slope*), serta tingkat kepanikan pasar. Pembagian model bias ke dalam tiga lapis rezim volatilitas (VIX Low, VIX Med, VIX High) didasarkan pada fakta empiris bahwa struktur kesalahan prediksi model tren pada kondisi pasar tenang sangat berbeda dibandingkan saat pasar mengalami krisis (VIX > 20.0).

## 2.7 Ensemble Learning dalam Peramalan Finansial (ML-ML & ML-DL Hybrid)
*Ensemble learning* menggabungkan beberapa model prediksi untuk menghasilkan satu prediksi akhir yang lebih kuat dengan varians yang lebih rendah (Dietterich, 2000). Penelitian ini mengeksplorasi dua jenis arsitektur ensemble:
1. **ML-ML Ensemble (Model C)**: Menggabungkan model pembelajaran mesin canggih (Two-Stage Decoupled Ridge) dengan model tren statis tanpa gating makro. Kombinasi ini bertindak sebagai regularisasi tambahan yang mereduksi varians prediksi lintasan jangka panjang secara linier.
2. **ML-DL Ensemble (Model D)**: Menggabungkan model linier teratur (Ridge) dengan model neural network non-linier (GRU). Struktur hybrid ini memanfaatkan keunggulan model Ridge dalam mempertahankan stabilitas tren linear jangka panjang dan keunggulan GRU dalam menangkap pola kejutan non-linier frekuensi tinggi dari guncangan makro harian.
