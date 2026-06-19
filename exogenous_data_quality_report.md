# Exogenous Data Quality Report

## Verdict
Saya tidak menemukan corruption keras seperti missing values, duplicate dates, atau nilai mustahil yang jelas.

Namun ada beberapa karakter data yang sangat penting karena bisa terlihat seperti “anomali” dalam model harian:

- series mixed-frequency yang di-forward-fill ke daily grid,
- level stepwise yang lama tidak berubah lalu lompat tajam,
- spike event yang jarang tetapi besar,
- distribution shift yang kuat antara train dan test.

## Hard Checks

- duplicate dates: `0`
- missing values: `0`
- train date gaps: hanya `1` dan `3` hari, sesuai business day pattern
- test date gaps: mayoritas `1` dan `3` hari, ada beberapa gap `2/4` hari karena kalender

## Variable-Level Notes

### `OIL`, `GOLD`, `SP500`, `IHSG`, `VIX`
- daily movement ada dan wajar
- spike besar muncul saat stress period
- `VIX` paling sering outlier secara rolling z-score
- ini bukan corruption; ini perilaku market event-driven

### `CPI`
- sangat stepwise
- hanya `14` unique values di train
- nilai berubah kira-kira setahun sekali
- max same-value run sekitar `261` hari

Ini terlihat seperti series annual / low-frequency yang di-forward-fill ke daily observations.

### `BI_rate`
- stepwise, tetapi lebih sering berubah daripada CPI
- `40` perubahan nilai di train
- max same-value run sekitar `392` hari
- update dates mengikuti keputusan kebijakan

### `US_rate`
- stepwise monthly-like series
- `118` perubahan nilai di train
- max same-value run sekitar `129` hari
- banyak perubahan kecil, lalu lompat tajam pada 2022-2023

## Anomaly-Like Behavior

### 1. Flat segments
`CPI`, `BI_rate`, dan `US_rate` banyak berada pada level yang sama selama hari-hari berturut-turut.

Ini bukan error per se, tetapi berarti:

- daily forecast model sering melihat fitur yang stale,
- informasi baru hanya datang pada event update days,
- hubungan same-day harian bisa lemah karena sebagian variabel sebenarnya low-frequency state.

### 2. Spike days
Contoh spike besar yang terlihat wajar secara ekonomi:

- `SP500` Maret 2020
- `VIX` Februari-Maret 2020
- `OIL` Maret-April 2020 dan Maret 2022
- `BI_rate` 2022-2023 tightening cycle
- `US_rate` 2022-2023 Fed hiking cycle

### 3. Train-test shift
Beberapa exogenous berubah distribusi kuat di test:

- `GOLD` mean train sekitar `1464` vs test sekitar `2978`
- `SP500` mean train sekitar `2452` vs test sekitar `5706`
- `US_rate` mean train sekitar `0.76` vs test sekitar `4.65`

Ini penting karena model yang belajar dari train lama bisa mengalami scale mismatch di test.

## What This Means For Modeling

Data exogenous tampaknya tidak rusak, tapi juga tidak homogen.

Yang paling mungkin mengganggu model harian adalah:

- mixed frequency masquerading as daily data,
- stale values yang panjang,
- sudden policy/event jumps,
- strong regime and scale shift.

## Recommended Handling

- treat CPI, BI_rate, US_rate as low-frequency state variables,
- add change flags / event-day indicators,
- use lagged changes, not just levels,
- consider scaling features within rolling windows or regimes,
- do not assume same-day daily predictability for every macro series.

## Bottom Line

Tidak ada bukti data exogenous ini “kotor” dalam arti salah input atau missing values.
Yang ada adalah struktur data yang **mixed-frequency, stepwise, dan event-driven**, sehingga kalau dipakai seperti daily continuous signals, model akan melihat banyak stale information dan beberapa spike yang sangat dominan.
