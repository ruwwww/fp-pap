# Exogenous Event Flag Investigation

## Hypothesis
Kalau exogenous memang mengandung signal yang tersembunyi, maka sinyal itu mungkin muncul bukan di level mentah, tetapi di:

- hari perubahan kebijakan / update data,
- event window di sekitar perubahan,
- state changes seperti `days_since_change`,
- spike flags untuk market variables.

## Features Tested

- `CPI_chg`, `BI_rate_chg`, `US_rate_chg`
- `chg_mag`
- `days_since_change`
- `evt_p1` and `evt_m1`
- spike flags untuk `OIL`, `GOLD`, `SP500`, `IHSG`, `VIX`

## Main Results

### 1. Event-day move test
Beberapa low-frequency variables memang berasosiasi dengan pergerakan USD/IDR yang lebih besar pada hari perubahan.

- `BI_rate` change day: mean abs move `87.03` vs `54.14` non-change, `p = 0.0426`
- `CPI` change day: mean abs move `88.57` vs `54.38` non-change, `p = 0.0894`
- `US_rate` change day: tidak signifikan, `p = 0.6556`

Ini menunjukkan ada event response, tapi tidak konsisten untuk semua variabel.

### 2. Walk-forward directional lift
Menambah change flags dan state features ke AR baseline tidak memberi lift yang stabil.

Contoh holdout terakhir:

- `AR5` AUC arah: `0.5861`
- `AR5 + EVENT features`: `0.5835`

Pada event window sekitar update days:

- `AR5` AUC: `0.5817`
- `AR5 + EVENT`: `0.5663`

Jadi change flags saja belum cukup untuk membuka signal yang robust.

### 3. Market spike flags
Spike flags untuk `OIL`, `GOLD`, `SP500`, `IHSG`, `VIX` juga tidak menunjukkan lift yang konsisten terhadap absolute move USD/IDR.

`VIX` paling dekat ke signal yang masuk akal, tetapi efeknya tetap lemah untuk menjadi feature utama.

## Interpretation

Kesimpulan investigasi langkah 2:

- change flags memang menangkap beberapa event response,
- tetapi sinyalnya terlalu sparse dan terlalu tidak stabil untuk menjadi predictor utama,
- jadi mereka lebih cocok dipakai sebagai **regime cue** atau **gating variable**, bukan direct forecasting signal.

## Practical Conclusion

Jika tujuan utama adalah mencari signal exogenous yang bisa dipakai model, maka event flags belum cukup.

Mereka hanya berguna kalau digabung dengan:

- lag structure yang learned,
- regime conditioning,
- dan interaction terms.

Tanpa itu, change flags menambah sedikit informasi, tetapi tidak cukup untuk mengalahkan autoregressive baseline secara konsisten.
