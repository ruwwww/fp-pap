import pandas as pd
import numpy as np
from pathlib import Path

def advanced_feature_engineering(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Menjalankan rekomendasi perbaikan kualitas data makroekonomi 
    berdasarkan hasil investigasi anomali.
    """
    out = df.copy()
    
    # Pastikan kolom tanggal berupa datetime
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 1. VIX CEILING ARTIFACT
    # Menghaluskan angka 41.6702 yang berulang di periode COVID
    # ---------------------------------------------------------
    if "VIX" in out.columns:
        # Cari mask di mana VIX persis di angka artifact itu (beri toleransi desimal)
        artifact_mask = np.isclose(out["VIX"], 41.6702, atol=1e-4)
        
        # Ubah jadi NaN sementara
        out.loc[artifact_mask, "VIX"] = np.nan
        
        # Karena ini data time-series, kita interpolate menggunakan polinomial orde 2 
        # (agar mengikuti bentuk lonjakan/kurva alami volatilitas, bukan garis lurus)
        # Jika polinomial gagal (karena jarak NaN terlalu besar), fallback ke linear
        try:
            out["VIX"] = out["VIX"].interpolate(method='polynomial', order=2)
        except:
            out["VIX"] = out["VIX"].interpolate(method='linear')
            
        out["VIX"] = out["VIX"].bfill().ffill()
        print(f"-> Diperbaiki {artifact_mask.sum()} baris artifact VIX.")

    # ---------------------------------------------------------
    # 2. CPI (STEP FUNCTION PROBLEM)
    # Mengubah CPI absolut tahunan menjadi Momentum Interpolasi
    # ---------------------------------------------------------
    if "CPI" in out.columns:
        # Cara terbaik: Buat garis halus di antara lompatan tahunan
        # Pertama, deteksi hari-hari di mana CPI berubah (biasanya awal tahun)
        cpi_change_mask = out["CPI"] != out["CPI"].shift(1)
        
        # Biarkan nilai yang berubah tetap ada, tapi jadikan NaN hari-hari yang "diam" (sama dengan hari sebelumnya)
        # TAPI pertahankan nilai pertama agar interpolasi punya titik awal
        out["CPI_Smoothed"] = out["CPI"]
        out.loc[(~cpi_change_mask) & (out.index > 0), "CPI_Smoothed"] = np.nan
        
        # Interpolasi linear untuk membuat tanjakan inflasi harian yang mulus
        out["CPI_Smoothed"] = out["CPI_Smoothed"].interpolate(method='linear')
        
        # Buat fitur Momentum (YoY atau MoM change dari data yang sudah di-smooth)
        out["CPI_Momentum"] = out["CPI_Smoothed"].pct_change(periods=21) * 100 # Perkiraan 21 hari kerja sebulan
        out["CPI_Momentum"] = out["CPI_Momentum"].fillna(0)
        
        print("-> CPI diubah menjadi CPI_Smoothed dan ditambahkan fitur CPI_Momentum.")

    # ---------------------------------------------------------
    # 3. INTEREST RATES (BI & US)
    # Mengelola step function dengan Spread dan Momentum
    # ---------------------------------------------------------
    if "BI_rate" in out.columns and "US_rate" in out.columns:
        # A. Suku Bunga Riil / Spread (Berapa beda cuan simpan uang di Indo vs US)
        out["Rate_Spread"] = out["BI_rate"] - out["US_rate"]
        
        # B. Detektor "Kejutan" Kebijakan (Apakah bulan ini ada kenaikan/penurunan?)
        out["BI_rate_Shock"] = out["BI_rate"].diff().fillna(0)
        out["US_rate_Shock"] = out["US_rate"].diff().fillna(0)
        
        # C. Momentum jangka panjang (Perubahan suku bunga dalam 3 bulan terakhir)
        out["Rate_Spread_Momentum_63d"] = out["Rate_Spread"] - out["Rate_Spread"].shift(63)
        out["Rate_Spread_Momentum_63d"] = out["Rate_Spread_Momentum_63d"].fillna(0)
        
        print("-> Fitur Rate_Spread, Shock, dan Momentum Suku Bunga ditambahkan.")

    # ---------------------------------------------------------
    # 4. COVID-19 PERIOD ISOLATION (REGIME FLAG)
    # ---------------------------------------------------------
    # Beri tanda "1" untuk periode kepanikan maksimal (Maret 2020 - Des 2020)
    covid_mask = (out[date_col] >= "2020-03-01") & (out[date_col] <= "2020-12-31")
    out["Regime_Crisis_COVID"] = covid_mask.astype(float)
    
    # Opsional: Deteksi krisis otomatis menggunakan VIX
    # Jika VIX > persentil 90, itu adalah masa krisis/panik.
    if "VIX" in out.columns:
        vix_p90 = out["VIX"].quantile(0.90)
        out["Regime_High_Vol"] = (out["VIX"] > vix_p90).astype(float)
        print("-> Fitur Regime Crisis (COVID & High Volatility) ditambahkan.")

    return out

# Contoh cara pemakaian
if __name__ == "__main__":
    # Load data yang sudah bersih dari anomali ekstrem (data_train_clean.csv)
    input_file = Path("data_train_clean.csv") 
    output_file = Path("data_train_engineered.csv")
    
    if input_file.exists():
        print(f"Memproses file: {input_file}")
        df = pd.read_csv(input_file)
        
        df_engineered = advanced_feature_engineering(df, date_col="Date")
        
        df_engineered.to_csv(output_file, index=False)
        print(f"\n[SUKSES] Dataset dengan fitur makro lanjutan disimpan ke: {output_file}")
    else:
        print(f"File {input_file} tidak ditemukan. Silakan sesuaikan nama file.")