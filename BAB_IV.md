# BAB IV. HASIL DAN PEMBAHASAN

## 4.1 Hasil Model A (Two-Stage Decoupled Ridge Model)
Model A dikonfigurasi menggunakan parameter optimal hasil *grid search cross-validation* di data training: `vix_factor = 1.05`, `spread_factor = 1.02`, dan `beta = 0.25`. Model ini dilatih pada 100% data training dan dievaluasi pada data test out-of-sample (Juni 2023 hingga pertengahan 2026).

Berikut adalah performa metrik evaluasi akhir dari Model A pada fase pengujian OOS:
* **Root Mean Squared Error (RMSE)**: **272.5646**
* **Mean Absolute Error (MAE)**: **217.3198**
* **Mean Absolute Percentage Error (MAPE)**: **1.3432%**
* **Koefisien Determinasi ($R^2$)**: **0.8010**

*Pembahasan*: Hasil ini sangat memuaskan secara akademis. Dengan nilai MAPE hanya sebesar **1.34%**, tingkat penyimpangan prediksi harian model berada di tingkat yang sangat kecil. Koefisien $R^2$ sebesar **0.8010** menandakan bahwa 80.1% variabilitas pergerakan rupiah yang volatil pada periode uji berhasil diproyeksikan dengan tepat oleh kombinasi model tren Ridge dan kejutan makroekonomi teratur ini.

---

## 4.2 Hasil Model B (Deep GRU Model)
Model B (Ridge Trend + GRU Residual) diuji secara rekursif menggunakan struktur jaringan GRU yang sangat diregulasi L2 keras. Evaluasi dilakukan pada data test out-of-sample yang sama.

Metrik evaluasi akhir dari Model B adalah:
* **Root Mean Squared Error (RMSE)**: **838.5002**
* **Mean Absolute Error (MAE)**: **639.6592**
* **Mean Absolute Percentage Error (MAPE)**: **3.9124%**
* **Koefisien Determinasi ($R^2$)**: **-0.8837**

*Pembahasan*: Model B menghasilkan prediksi yang stabil tanpa ada efek *prediction explosion* (nilai tak terhingga). Namun, akurasi model ini menurun drastis dibandingkan Model A. Nilai $R^2$ negatif menandakan model GRU memberikan hasil prediksi yang lebih buruk dibandingkan nilai rata-rata historis sederhana. Hal ini terjadi karena model deep learning sangat sensitif terhadap *noise* data keuangan harian. Pada peramalan rekursif jangka panjang, kesalahan kecil non-linier dari output GRU terakumulasi di setiap langkah peramalan, sehingga menyebabkan deviasi arah prediksi yang signifikan di akhir horizon.

---

## 4.3 Hasil Model C (ML-ML Ensemble)
Model C menggabungkan model terbaik (Model A) dengan model tren statis tanpa gating makro dan koreksi bias.
Metrik evaluasi akhir Model C pada pengujian OOS adalah:
* **Root Mean Squared Error (RMSE)**: **271.8921**
* **Mean Absolute Error (MAE)**: **209.1605**
* **Mean Absolute Percentage Error (MAPE)**: **1.2961%**
* **Koefisien Determinasi ($R^2$)**: **0.8019**

*Pembahasan*: Model C memberikan performa **terbaik secara absolut** di antara seluruh model. Dengan mencampurkan model linier aktif (Model A) dengan baseline tren statis yang halus, ensemble ML-ML ini memperoleh keuntungan berupa penurunan varians model (*variance reduction*). Rata-rata linier dari kedua prediksi ML ini bertindak sebagai peredam volatilitas berlebih, menghasilkan lintasan prediksi yang lebih akurat dengan RMSE terendah **271.89**.

---

## 4.4 Hasil Model D (ML-DL Ensemble)
Model D menggabungkan Model A (Machine Learning Ridge) dengan Model B (Deep Learning GRU).
Metrik evaluasi akhir Model D pada pengujian OOS adalah:
* **Root Mean Squared Error (RMSE)**: **433.8867**
* **Mean Absolute Error (MAE)**: **352.6844**
* **Mean Absolute Percentage Error (MAPE)**: **2.1701%**
* **Koefisien Determinasi ($R^2$)**: **0.4956**

*Pembahasan*: Struktur hybrid ML-DL ini menunjukkan performa yang cukup kuat dengan nilai $R^2$ positif sebesar **0.4956** dan RMSE **433.89**. Model D berhasil meredam ketidakstabilan model GRU murni (Model B) berkat kontribusi 50% dari kestabilan model Ridge (Model A), menjadikannya alternatif yang layak jika integrasi kecerdasan buatan dan statistik tradisional diwajibkan dalam sistem.

---

## 4.5 Perbandingan Hasil Evaluasi Akhir (OOS Test Set)
Tabel di bawah merangkum performa dari keempat model yang diuji secara out-of-sample pada data uji sesungguhnya:

| Kategori Model | Model Spesifik | RMSE | MAE | MAPE (%) | $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A (Machine Learning)** | Two-Stage Decoupled Ridge | **272.5646** | 217.3198 | 1.3432% | 0.800958 |
| **Model B (Deep Learning)** | Ridge Trend + Deep GRU | **838.5002** | 639.6592 | 3.912362% | -0.883700 |
| **Model C (Ensemble ML-ML)** | Decoupled Ridge + Static Trend | **271.8921** | **209.1605** | **1.296130%** | **0.801939** |
| **Model D (Ensemble ML-DL)** | Decoupled Ridge + Deep GRU | **433.8867** | 352.6844 | 2.170135% | 0.495620 |

---

## 4.6 Analisis Kelebihan dan Kekurangan Berbasis Ekonometrika dan Makroekonomi
Perbedaan kinerja yang mencolok antara model tradisional Machine Learning (Model A dan C) dengan Deep Learning (Model B) dapat dijelaskan melalui analisis metodologis dan teori ekonomi:

1. **Stabilitas Model Linier Terregulasi**:
   Kurs USDIDR dipengaruhi oleh tren jangka panjang makroekonomi Indonesia (seperti inflasi relatif dan tingkat pertumbuhan ekonomi). Model Ridge dengan regulasi L2 berhasil mengestimasi parameter tren secara stabil tanpa mengalami *overfitting* pada anomali jangka pendek. Ini memberi Model A landasan prediksi yang sangat kuat.
2. **Kekurangan Deep Learning pada Peramalan Rekursif**:
   Meskipun GRU sangat canggih dalam mengenali pola non-linier kompleks, ia membutuhkan data dalam jumlah raksasa untuk generalisasi yang baik. Pada runtun waktu keuangan harian yang sarat derau acak (*random walk features*), GRU cenderung menangkap *noise* sebagai sinyal nyata. Saat meramal 778 langkah ke depan secara rekursif, eror minor di awal horizon terakumulasi secara berantai, mendistorsi prediksi akhir. Oleh karena itu, performa GRU murni (Model B) menjadi kurang optimal.
3. **Efektivitas Gerbang Makro Dinamis**:
   Penerapan aturan gating (VIX > 14.0 dan Spread < 0.8%) terbukti secara empiris menaikkan akurasi model A dan C. Saat volatilitas global tinggi (VIX tinggi) dan spread menyempit (BI Rate kurang kompetitif), rupiah secara nyata mengalami tekanan depresiasi akibat penarikan dana investor asing (*capital flight*). Menangkap fenomena asimetris ini secara langsung menggunakan aturan gating ekonomi terbukti jauh lebih efektif dibanding menyerahkan pencarian pola tersebut sepenuhnya kepada sel neural network GRU yang minim data pelatihan.

---

## 4.7 Visualisasi Lintasan Prediksi Final
Visualisasi lintasan peramalan jangka panjang out-of-sample dari keempat model dibandingkan dengan data aktual disajikan secara grafis pada grafik perbandingan visual **`submission_predictions_plot.png`** yang dihasilkan oleh script [colab_notebook.py](file:///C:/kuliahh%20maseh/pap/eas/colab_notebook.py). 

Grafik tersebut menunjukkan bahwa Model A dan Model C memiliki lintasan prediksi yang menempel sangat dekat dengan data aktual USDIDR selama 778 hari masa pengujian, sementara Model D dan Model B mengalami deviasi pelemahan yang lebih lebar namun tetap terkontrol di bawah rentang Rp17.500 per USD. Hasil peramalan terbaik dari Model A disimpan ke dalam file **[submission.csv](file:///C:/kuliahh%20maseh/pap/eas/submission.csv)** untuk diunggah pada kompetisi.
