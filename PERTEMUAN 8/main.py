import logging
from pathlib import Path

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf

# KONFIGURASI DAN KONSTANTA

# Path file input dan output
FILE_INPUT = Path("./DATA/student_exam_scores.xlsx")
FILE_OUTPUT = Path("./DATA/student_exam_scores_cleaned.csv")

# Kolom numerik untuk analisis
KOLOM_NUMERIK = ["hours_studied", "sleep_hours", "attendance_percent", "exam_score"]

# Variabel prediktor dan target
PREDIKTOR = ["hours_studied", "sleep_hours", "attendance_percent"]
TARGET = "exam_score"

# Level signifikansi untuk uji statistik
ALPHA = 0.05

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# FUNGSI UNTUK MEMUAT DAN MENYIMPAN DATA

def muat_data(path: Path) -> pd.DataFrame:
    """
    Memuat data dari file Excel.

    Args:
        path: Path ke file Excel

    Returns:
        DataFrame berisi data yang dimuat
    """
    logging.info(f"Memuat data dari {path}")
    df = pd.read_excel(path)
    logging.info(f"Data berhasil dimuat. Ukuran: {df.shape}")
    return df


def simpan_data_bersih(df: pd.DataFrame, path: Path) -> None:
    """
    Menyimpan DataFrame yang sudah dibersihkan ke file CSV.

    Args:
        df: DataFrame yang akan disimpan
        path: Path tujuan file CSV
    """
    logging.info(f"Menyimpan data bersih ke {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Data berhasil disimpan")


# FUNGSI UNTUK EKSPLORASI DATA

def tampilkan_info_dataset(df: pd.DataFrame) -> None:
    """
    Menampilkan informasi dasar tentang dataset.

    Args:
        df: DataFrame yang akan dianalisis
    """
    print("\n" + "=" * 80)
    print("INFORMASI DATASET")
    print("=" * 80)

    # Ukuran dataset
    print(f"\nUkuran Dataset: {df.shape[0]} baris x {df.shape[1]} kolom")

    # Informasi tipe data
    print("\nInformasi Tipe Data:")
    df.info()

    # Missing values
    print("\nJumlah Missing Values per Kolom:")
    print(df.isnull().sum())

    # Duplicate rows
    duplikat = df.duplicated().sum()
    print(f"\nJumlah Baris Duplikat: {duplikat}")

    # Preview data
    print("\n5 Baris Pertama:")
    print(df.head())


def tampilkan_statistik_deskriptif(df: pd.DataFrame) -> None:
    """
    Menampilkan statistik deskriptif dari dataset.

    Args:
        df: DataFrame yang akan dianalisis
    """
    print("\n" + "=" * 80)
    print("STATISTIK DESKRIPTIF")
    print("=" * 80)
    print(df.describe())


# FUNGSI UNTUK PEMBERSIHAN DATA

def bersihkan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan dataset dengan:
    - Menghapus baris duplikat
    - Mengisi nilai missing dengan mean kolom

    Args:
        df: DataFrame yang akan dibersihkan

    Returns:
        DataFrame yang sudah dibersihkan
    """
    logging.info("Memulai pembersihan data...")

    # Hapus duplikat
    df_bersih = df.drop_duplicates().copy()
    logging.info(f"Menghapus {df.shape[0] - df_bersih.shape[0]} baris duplikat")

    # Isi missing values dengan mean untuk kolom numerik
    for col in df_bersih.columns:
        if pd.api.types.is_numeric_dtype(df_bersih[col]) and df_bersih[col].isnull().any():
            mean_val = df_bersih[col].mean()
            df_bersih[col] = df_bersih[col].fillna(mean_val)
            logging.info(
                f"Mengisi {df[col].isnull().sum()} nilai missing pada kolom '{col}' dengan mean ({mean_val:.2f})")

    logging.info(f"Data setelah pembersihan: {df_bersih.shape}")
    return df_bersih


# FUNGSI UNTUK DETEKSI OUTLIER

def plot_boxplot_outliers(df: pd.DataFrame, kolom: List[str]) -> None:
    """
    Membuat boxplot untuk mendeteksi outlier pada kolom numerik.

    Args:
        df: DataFrame yang akan divisualisasikan
        kolom: List kolom numerik untuk dianalisis
    """
    logging.info("Membuat boxplot untuk deteksi outlier...")

    plt.figure(figsize=(12, 5))

    for i, col in enumerate(kolom, start=1):
        plt.subplot(1, len(kolom), i)
        sns.boxplot(data=df, y=col, color='skyblue')
        plt.title(f"Boxplot: {col}", fontsize=10, fontweight='bold')
        plt.ylabel(col)

    plt.tight_layout()
    plt.show()


# FUNGSI UNTUK ANALISIS KORELASI

def plot_correlation_heatmap(df: pd.DataFrame, kolom: List[str], target: str) -> None:
    """
    Membuat heatmap matriks korelasi.

    Args:
        df: DataFrame yang akan dianalisis
        kolom: List kolom prediktor
        target: Kolom target
    """
    logging.info("Membuat heatmap korelasi...")

    # Hitung matriks korelasi
    correlation_matrix = df[kolom + [target]].corr()

    print("\n" + "=" * 80)
    print("MATRIKS KORELASI")
    print("=" * 80)
    print(correlation_matrix)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriks Korelasi Antar Variabel", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


# FUNGSI UNTUK VISUALISASI SCATTER PLOT

def plot_scatter_plots(df: pd.DataFrame, kolom: List[str], target: str) -> None:
    """
    Membuat scatter plot untuk setiap prediktor vs target.

    Args:
        df: DataFrame yang akan divisualisasikan
        kolom: List kolom prediktor
        target: Kolom target
    """
    logging.info("Membuat scatter plots...")

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(kolom, start=1):
        plt.subplot(1, len(kolom), i)
        sns.scatterplot(data=df, x=col, y=target, alpha=0.6, color='steelblue')
        plt.title(f"{col} vs {target}", fontsize=10, fontweight='bold')
        plt.xlabel(col)
        plt.ylabel(target)

        # Tambahkan garis tren
        z = np.polyfit(df[col], df[target], 1)
        p = np.poly1d(z)
        plt.plot(df[col], p(df[col]), "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.show()


# FUNGSI UNTUK UJI NORMALITAS

def uji_normalitas_shapiro(df: pd.DataFrame, kolom: str, alpha: float = ALPHA) -> None:
    """
    Melakukan uji normalitas Shapiro-Wilk dan membuat histogram.

    Args:
        df: DataFrame yang akan diuji
        kolom: Nama kolom yang akan diuji normalitasnya
        alpha: Level signifikansi (default: 0.05)
    """
    logging.info(f"Melakukan uji normalitas Shapiro-Wilk pada kolom '{kolom}'...")

    # Lakukan uji Shapiro-Wilk
    stat, pvalue = stats.shapiro(df[kolom])

    print("\n" + "=" * 80)
    print("UJI NORMALITAS SHAPIRO-WILK")
    print("=" * 80)
    print(f"Kolom yang diuji: {kolom}")
    print(f"Statistik W: {stat:.6f}")
    print(f"P-value: {pvalue:.6f}")
    print(f"Alpha: {alpha}")

    # Interpretasi hasil
    if pvalue > alpha:
        print(f"\nKesimpulan: Data '{kolom}' MENGIKUTI distribusi normal")
        print("(Gagal menolak H0 karena p-value > alpha)")
    else:
        print(f"\nKesimpulan: Data '{kolom}' TIDAK MENGIKUTI distribusi normal")
        print("(Tolak H0 karena p-value ≤ alpha)")

    # Plot histogram dengan kurva KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(df[kolom], kde=True, bins=20, color='steelblue', edgecolor='black')
    plt.title(f"Distribusi {kolom}", fontsize=14, fontweight='bold')
    plt.xlabel(kolom)
    plt.ylabel("Frekuensi")
    plt.axvline(df[kolom].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(df[kolom].median(), color='green', linestyle='--', linewidth=2, label='Median')
    plt.legend()
    plt.tight_layout()
    plt.show()


# FUNGSI UNTUK REGRESI LINEAR

def buat_model_regresi(df: pd.DataFrame, prediktor: str, target: str):
    """
    Membuat model regresi linear sederhana dan menampilkan hasilnya.

    Args:
        df: DataFrame yang berisi data
        prediktor: Nama kolom prediktor
        target: Nama kolom target

    Returns:
        Model regresi yang sudah di-fit
    """
    logging.info(f"Membuat model regresi: {target} ~ {prediktor}")

    # Fit model OLS
    formula = f"{target} ~ {prediktor}"
    model = smf.ols(formula, data=df).fit()

    # Ekstrak parameter
    intercept = model.params["Intercept"]
    slope = model.params[prediktor]
    ssr = model.ssr
    r_squared = model.rsquared

    # Tampilkan hasil
    print("\n" + "=" * 80)
    print("HASIL REGRESI LINEAR SEDERHANA")
    print("=" * 80)
    print(f"Model: {target} = β₀ + β₁ × {prediktor}")
    print(f"\nParameter:")
    print(f"  - Intercept (β₀): {intercept:.4f}")
    print(f"  - Slope (β₁): {slope:.4f}")
    print(f"\nPersamaan Regresi:")
    print(f"  {target} = {intercept:.4f} + {slope:.4f} × {prediktor}")
    print(f"\nGoodness of Fit:")
    print(f"  - R² (Koefisien Determinasi): {r_squared:.4f} ({r_squared * 100:.2f}%)")
    print(f"  - SSR (Sum of Squared Residuals): {ssr:.4f}")
    print(f"\nInterpretasi:")
    print(f"  Setiap kenaikan 1 unit pada {prediktor}, {target} akan")
    print(f"  {'naik' if slope > 0 else 'turun'} sebesar {abs(slope):.4f} unit")
    print(f"  Model menjelaskan {r_squared * 100:.2f}% variasi dalam {target}")

    return model


# FUNGSI UTAMA

def main():
    """
    Fungsi utama yang mengatur alur eksekusi program.
    """
    print("\n" + "=" * 80)
    print("ANALISIS DATA NILAI UJIAN MAHASISWA")
    print("=" * 80)

    # 1. MUAT DATA
    df = muat_data(FILE_INPUT)

    # 2. EKSPLORASI DATA AWAL
    tampilkan_info_dataset(df)

    # 3. DETEKSI OUTLIER
    plot_boxplot_outliers(df, KOLOM_NUMERIK)

    # 4. PEMBERSIHAN DATA
    df_bersih = bersihkan_data(df)

    print("\n" + "=" * 80)
    print("PEMBERSIHAN DATA")
    print("=" * 80)
    print(f"Jumlah data setelah pembersihan: {df_bersih.shape}")

    # 5. SIMPAN DATA BERSIH
    simpan_data_bersih(df_bersih, FILE_OUTPUT)

    # 6. STATISTIK DESKRIPTIF
    tampilkan_statistik_deskriptif(df_bersih)

    # 7. ANALISIS KORELASI
    plot_correlation_heatmap(df_bersih, PREDIKTOR, TARGET)

    # 8. VISUALISASI HUBUNGAN ANTAR VARIABEL
    plot_scatter_plots(df_bersih, PREDIKTOR, TARGET)

    # 9. UJI NORMALITAS
    uji_normalitas_shapiro(df_bersih, TARGET)

    # 10. REGRESI LINEAR
    buat_model_regresi(df_bersih, "hours_studied", TARGET)

    print("\n" + "=" * 80)
    print("ANALISIS SELESAI")
    print("=" * 80)


# EKSEKUSI PROGRAM

if __name__ == "__main__":
    main()
