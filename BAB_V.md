# BAB V. KESIMPULAN DAN SARAN

## 5.1 Kesimpulan
Secara umum, kesimpulan dari penelitian ini dapat disintesiskan ke dalam tiga pandangan konseptual yang lebih luas mengenai peramalan nilai tukar dalam ekonomi makro:
1. **Batas Kompleksitas Komputasi dalam Dinamika Pasar yang Kaotik**:
   Pergerakan nilai tukar harian merupakan cerminan dari ekspektasi psikologis pelaku pasar, kebijakan moneter, dan risiko geopolitik yang sangat fluktuatif. Eksperimen ini menunjukkan bahwa dalam menghadapi runtun waktu finansial yang sarat akan ketidakpastian (*entropy*), penggunaan arsitektur komputasi yang terlalu kompleks dan fleksibel justru rentan menangkap derau acak sebagai sebuah kepastian. Sebaliknya, pendekatan terstruktur yang mengutamakan penyederhanaan teratur terbukti memberikan konsistensi arah peramalan yang jauh lebih andal untuk jangka panjang.
2. **Sinergi Pengetahuan Struktural (Ekonomi) dan Sains Data**:
   Sains data prediktif tidak dapat berdiri sendiri sebagai sekumpulan algoritma yang sepenuhnya digerakkan oleh data (*purely data-driven*). Hasil penelitian ini membuktikan bahwa integrasi teori ekonomi makro—seperti perilaku asimetris aliran modal akibat tingkat kecemasan global dan daya tarik imbal hasil domestik—bertindak sebagai koridor rasional bagi algoritma. Pengetahuan domain ini memberikan "kompas arah" yang menjaga agar model prediktif tidak menghasilkan proyeksi lintasan yang bertentangan dengan realitas fundamental ekonomi.
3. **Kestabilan Jangka Panjang sebagai Prioritas Keputusan**:
   Dalam perspektif pengambilan kebijakan strategis, kegunaan utama dari model peramalan bukanlah untuk menangkap fluktuasi harian mikro secara presisi, melainkan untuk memberikan gambaran lintasan tren jangka panjang yang stabil dan dapat dipertanggungjawabkan fundamentalnya. Pemisahan sinyal makro dan mikro, dikombinasikan dengan mekanisme koreksi bias berkelanjutan, memastikan bahwa model tetap kokoh dalam menyajikan proyeksi nilai tukar yang realistis bagi perencanaan makroekonomi jangka panjang.

## 5.2 Saran
Beberapa saran yang direkomendasikan untuk pengembangan penelitian selanjutnya adalah:
1. **Pengembangan Gerbang Kontinu (Smooth Gating)**:
   Disarankan untuk mengganti aturan biner gerbang makroekonomi (*hard threshold*) menjadi fungsi pembobotan kontinu (seperti fungsi sigmoid atau kurva logistik) agar akselerasi return total bertransisi secara halus ketika mendekati titik kritis (misalnya saat VIX mendekati 14.0).
2. **Penambahan Indikator Struktural Jangka Menengah**:
   Penelitian selanjutnya dapat mengeksplorasi penambahan variabel eksternal makro Indonesia lainnya, seperti data rilis bulanan Neraca Perdagangan, Cadangan Devisa, dan inflasi relatif (CPI selisih) sebagai kontrol terhadap pergeseran tingkat struktural nilai tukar dalam jangka menengah.
3. **Eksplorasi Arsitektur Deep Learning Khusus Runtun Waktu**:
   Untuk meningkatkan performa Deep Learning pada peramalan rekursif, dapat dicoba arsitektur mutakhir seperti PatchTST (*Patch Time Series Transformer*) atau TFT (*Temporal Fusion Transformers*) dengan ukuran window latih yang disesuaikan secara lokal untuk mencegah penumpukan eror rekursif.
