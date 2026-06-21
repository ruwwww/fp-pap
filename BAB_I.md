# BAB I. PENDAHULUAN

## 1.1 Latar Belakang
Analisis runtun waktu (*time series analysis*) merupakan salah satu pilar penting dalam pemodelan kuantitatif keuangan dan ekonomi prediktif. Karakteristik utama dari data runtun waktu keuangan adalah pergerakannya yang dinamis, non-stasioner, dan sering kali dipengaruhi oleh guncangan makroekonomi harian (Shumway & Stoffer, 2019). Salah satu variabel keuangan yang memiliki urgensi tinggi untuk diamati dan diprediksi adalah nilai tukar Rupiah terhadap Dollar Amerika Serikat (USDIDR). 

Pergerakan nilai tukar USDIDR memegang peranan krusial dalam kestabilan ekonomi makro Indonesia. Fluktuasi kurs memengaruhi harga barang impor (*imported inflation*), neraca perdagangan, daya saing ekspor, cadangan devisa, hingga kesanggupan pemerintah dan korporasi dalam membayar beban utang luar negeri. Oleh karena itu, peramalan kurs USDIDR jangka panjang yang akurat sangat dibutuhkan oleh otoritas moneter seperti Bank Indonesia untuk merancang intervensi pasar yang tepat, serta oleh para pelaku bisnis untuk melakukan lindung nilai (*hedging*) portofolio keuangan mereka.

Namun, melakukan peramalan jangka panjang (*multi-step ahead recursive forecasting*) pada kurs harian USDIDR memiliki tantangan ekonometrika yang sangat berat. Model statistik tradisional seperti ARIMA cenderung menghasilkan prediksi yang terlalu mulus atau datar (*mean-damped*) saat melakukan peramalan rekursif jangka panjang, karena proses rekursif menyaring seluruh kejutan volatilitas harian. Di sisi lain, model pembelajaran mendalam (*deep learning*) non-linier kompleks seperti LSTM dan GRU sangat rentan terhadap *overfitting* dan ketidakstabilan numerik saat dihadapkan pada data keuangan yang sarat akan derau acak (*random walk noise*).

Untuk menjembatani keterbatasan ini, penelitian praktikum ini mengajukan arsitektur **Two-Stage Decoupled Ridge Model** yang terintegrasi dengan akselerasi gerbang makroekonomi (*dynamic macro gating*) dan sistem koreksi bias dinamis (*volatility-regime bias correction*). Pendekatan ini memisahkan sinyal tren jangka panjang yang diestimasi menggunakan lag autoregresif dari fluktuasi harian yang disebabkan oleh kejutan makro eksternal. Model ini memanfaatkan variabel makroekonomi utama seperti volatilitas pasar global (Indeks VIX) dan selisih suku bunga kebijakan moneter (Spread BI-US Rate) sebagai akselerator gerbang asimetris untuk memodelkan pelemahan Rupiah saat terjadi aliran modal keluar (*capital flight*).

## 1.2 Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian praktikum ini adalah sebagai berikut:
1. Bagaimana merancang arsitektur model terdekopel dua-tahap (*Two-Stage Decoupled Ridge*) yang mampu memisahkan komponen tren jangka panjang dan kejutan makro harian pada peramalan kurs USDIDR secara rekursif jangka panjang?
2. Bagaimana pengaruh integrasi gerbang akselerasi makroekonomi berbasis Indeks VIX dan selisih suku bunga (Spread BI-US Rate) terhadap akurasi peramalan out-of-sample USDIDR?
3. Bagaimana perbandingan performa peramalan out-of-sample antara model Machine Learning tradisional teratur (Model A: Decoupled Ridge), model Deep Learning (Model B: Deep GRU), serta model gabungan (Model C: ML-ML Ensemble dan Model D: ML-DL Ensemble) berdasarkan metrik evaluasi RMSE, MAE, MAPE, dan $R^2$?

## 1.3 Tujuan
Tujuan yang ingin dicapai dari pelaksanaan penelitian praktikum ini adalah:
1. Membangun dan menguji keandalan arsitektur *Two-Stage Decoupled Ridge Model* dengan sistem koreksi bias berbasis rezim volatilitas untuk meramal lintasan USDIDR harian secara rekursif tanpa kebocoran data (*zero leakage*).
2. Menganalisis efektivitas penggabungan aturan ekonomi makro berupa gerbang akselerasi asimetris (VIX dan Spread Suku Bunga) dalam menangkap perilaku depresiasi Rupiah saat terjadi guncangan pasar keuangan.
3. Mengevaluasi dan membandingkan performa peramalan out-of-sample dari keempat variasi model (Model A, B, C, dan D) untuk menentukan model terbaik yang memiliki stabilitas dan akurasi paling optimal.

## 1.4 Manfaat
Penelitian praktikum ini diharapkan memberikan kontribusi nyata bagi berbagai pihak:

### 1.4.1 Bagi Mahasiswa
1. Memberikan pemahaman praktis yang mendalam mengenai penerapan teori runtun waktu keuangan (*financial time series*) dan pemodelan prediktif pada data dunia nyata.
2. Mengembangkan keterampilan teknis dalam rekayasa fitur kausal, implementasi algoritma regularisasi L2 (Ridge), arsitektur deep learning sekuensial (GRU), dan perancangan *ensemble learning* menggunakan Python dan TensorFlow.
3. Melatih kemampuan berpikir kritis dalam mengaitkan fenomena ekonomi makro riil (seperti pengetatan moneter Fed dan sentimen *risk-off*) dengan kinerja model matematika-statistik.

### 1.4.2 Bagi Institut
1. Menambah dokumentasi studi kasus dan portofolio penelitian terapan mahasiswa di Departemen Sistem Informasi, Institut Teknologi Sepuluh Nopember, khususnya di bidang analitika prediktif dan sains data keuangan.
2. Menyediakan materi rujukan akademis yang menunjukkan integrasi yang sukses antara kaidah machine learning yang ketat (seperti *walk-forward cross-validation*) dengan prinsip-prinsip ekonomi empiris.

### 1.4.3 Bagi Masyarakat
1. Menyediakan alternatif model proyeksi nilai tukar USDIDR yang transparan, dapat dipertanggungjawabkan secara akademis, dan bebas dari bias subjektif manusia.
2. Memberikan wawasan edukasi mengenai faktor-faktor pemicu pelemahan Rupiah, seperti penyempitan selisih suku bunga domestik dan lonjakan kecemasan pasar global, sehingga masyarakat dan pelaku usaha kecil menengah (UKM) dapat mengantisipasi risiko fluktuasi kurs secara lebih adaptif.
