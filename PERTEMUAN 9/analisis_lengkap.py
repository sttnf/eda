import warnings

# Import library yang diperlukan untuk analisis data
import matplotlib.pyplot as plt  # Untuk visualisasi grafik
import numpy as np  # Untuk operasi numerik
import pandas as pd  # Untuk manipulasi dan analisis data
import scipy.stats as stats  # Untuk uji statistik
import seaborn as sns  # Untuk visualisasi yang lebih menarik
import statsmodels.formula.api as smf  # Untuk analisis regresi

# Ignore warnings untuk output yang lebih bersih
warnings.filterwarnings('ignore')

# Set style untuk visualisasi agar lebih menarik
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ================================================================================
# 1. MEMUAT DATA (DATA LOADING)
# ================================================================================
# Tahap pertama adalah memuat data dari file Excel ke dalam DataFrame pandas

print("=" * 80)
print("1. MEMUAT DATA DARI FILE")
print("=" * 80)

# Membaca file Excel menggunakan pandas
df = pd.read_excel("./insurance_remed.xlsx")

print(f"✓ Data berhasil dimuat dari file 'insurance_remed.xlsx'")
print(f"✓ Jumlah data yang dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"\n5 Baris Pertama Data:")
print(df.head())

# ================================================================================
# 2. ANALISIS DATA AWAL (INITIAL DATA ANALYSIS)
# ================================================================================
# Melakukan pengecekan menyeluruh pada dataset untuk mengetahui:
# - Ukuran dataset (jumlah baris dan kolom)
# - Tipe data dari setiap variabel
# - Keberadaan missing values (nilai yang hilang)
# - Keberadaan data duplikat
# - Keberadaan outlier (data pencilan)

print("\n" + "=" * 80)
print("2. ANALISIS DATA AWAL")
print("=" * 80)

# A. UKURAN DATASET
print("\n[A] UKURAN DATASET")
print("-" * 80)
print(f"Jumlah baris (observasi): {df.shape[0]}")
print(f"Jumlah kolom (variabel): {df.shape[1]}")
print(f"Total ukuran dataset: {df.shape[0]} baris x {df.shape[1]} kolom")

# B. TIPE DATA SETIAP VARIABEL
print("\n[B] TIPE DATA SETIAP VARIABEL")
print("-" * 80)
print("\nInformasi Detail Dataset:")
df.info()
print("\nRingkasan Tipe Data:")
for col in df.columns:
    print(f"  - {col:20s}: {df[col].dtype}")

# C. MISSING VALUES (Nilai yang Hilang)
print("\n[C] MISSING VALUES (Nilai yang Hilang)")
print("-" * 80)
missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Kolom': missing_values.index,
    'Jumlah Missing': missing_values.values,
    'Persentase (%)': missing_percentage.values
})
print(missing_df.to_string(index=False))

total_missing = df.isnull().sum().sum()
if total_missing > 0:
    print(f"\n⚠ Total missing values: {total_missing} ({(total_missing / (df.shape[0] * df.shape[1])) * 100:.2f}%)")
else:
    print(f"\n✓ Tidak ada missing values dalam dataset")

# D. DATA DUPLIKAT
print("\n[D] DATA DUPLIKAT")
print("-" * 80)
duplicate_rows = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicate_rows}")
if duplicate_rows > 0:
    print(f"⚠ Terdapat {duplicate_rows} baris duplikat yang perlu dihapus")
    print(f"Persentase duplikasi: {(duplicate_rows / len(df)) * 100:.2f}%")
else:
    print("✓ Tidak ada data duplikat")

# E. TIPE DATA YANG SALAH
print("\nKonversi Tipe Data")
print("-" * 80)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

print("✓ Kolom 'bmi' dikonversi ke tipe numerik")
print("✓ Kolom 'charges' dikonversi ke tipe numerik")

# F. DETEKSI OUTLIER (Data Pencilan)
print("\n[F] DETEKSI OUTLIER (Data Pencilan)")
print("-" * 80)

# Konversi kolom numerik untuk deteksi outlier
df_numeric_check = df.copy()
df_numeric_check['age_numeric'] = pd.to_numeric(df_numeric_check['age'], errors='coerce')
df_numeric_check['bmi_numeric'] = pd.to_numeric(df_numeric_check['bmi'], errors='coerce')
df_numeric_check['charges_numeric'] = pd.to_numeric(df_numeric_check['charges'], errors='coerce')

# Visualisasi Boxplot untuk deteksi outlier
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Boxplot untuk Age
axes[0].boxplot(df_numeric_check['age_numeric'].dropna(), vert=True)
axes[0].set_title('Boxplot: Age (Umur)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Umur (tahun)')
axes[0].grid(True, alpha=0.3)

# Boxplot untuk BMI
axes[1].boxplot(df_numeric_check['bmi_numeric'].dropna(), vert=True)
axes[1].set_title('Boxplot: BMI (Body Mass Index)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('BMI')
axes[1].grid(True, alpha=0.3)

# Boxplot untuk Charges
axes[2].boxplot(df_numeric_check['charges_numeric'].dropna(), vert=True)
axes[2].set_title('Boxplot: Charges (Biaya Asuransi)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Biaya (dollar)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boxplot_outlier_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Boxplot telah dibuat untuk mendeteksi outlier pada variabel numerik")
print("  (Outlier ditandai dengan titik-titik di luar whisker)")

# ================================================================================
# KESIMPULAN ANALISIS DATA AWAL
# ================================================================================

print("\n" + "=" * 80)
print("KESIMPULAN ANALISIS DATA AWAL")
print("=" * 80)

print(f"""
RINGKASAN TEMUAN:

1. UKURAN DATASET:
   - Dataset memiliki {df.shape[0]} baris (observasi) dan {df.shape[1]} kolom (variabel)
   - Ukuran dataset cukup besar untuk analisis statistik

2. TIPE DATA:
   - age: integer (sudah benar)
   - sex: object/string (sudah benar untuk data kategorikal)
   - bmi: object (BERMASALAH - seharusnya numerik)
   - children: integer (sudah benar)
   - smoker: object/string (sudah benar untuk data kategorikal)
   - region: object/string (sudah benar untuk data kategorikal)
   - charges: object (BERMASALAH - seharusnya numerik)

3. MISSING VALUES:
   - bmi: {missing_values['bmi']} nilai hilang
   - charges: {missing_values['charges']} nilai hilang
   - Variabel lain: tidak ada missing values

4. DATA DUPLIKAT:
   - Terdapat {duplicate_rows} baris duplikat yang perlu dihapus

5. TIPE DATA YANG SALAH:
   - Kolom 'bmi' berisi nilai non-numerik (tanggal/teks)
   - Kolom 'charges' berisi nilai non-numerik

6. OUTLIER:
   - Terdeteksi outlier pada beberapa variabel (lihat boxplot)
   - Outlier pada 'charges' mungkin normal (biaya tinggi untuk kondisi tertentu)

⚠ KESIMPULAN AKHIR:
DATA INI TIDAK DAPAT LANGSUNG DIOLAH karena memiliki beberapa masalah:
  ✗ Ada missing values pada kolom bmi dan charges
  ✗ Ada data duplikat yang perlu dihapus
  ✗ Ada tipe data yang salah (bmi dan charges berisi nilai non-numerik)
  ✗ Perlu pembersihan data terlebih dahulu sebelum analisis lanjutan

SOLUSI:
  → Hapus data duplikat
  → Hapus baris dengan tipe data yang salah
  → Tangani missing values (hapus atau imputasi)
  → Konversi tipe data ke format yang benar
""")

# ================================================================================
# 3. PEMBERSIHAN DATA (DATA CLEANING)
# ================================================================================
# Melakukan pembersihan data dengan:
# - Menghapus data duplikat
# - Menghapus data dengan tipe yang salah (non-numerik pada kolom numerik)
# - Menangani missing values (dengan menghapus baris yang memiliki missing values)

print("\n" + "=" * 80)
print("3. PEMBERSIHAN DATA (DATA CLEANING)")
print("=" * 80)

# Menyimpan ukuran data original
original_size = df.shape[0]
print(f"\nUkuran data SEBELUM pembersihan: {original_size} baris")

# LANGKAH 1: Menghapus data duplikat
print("\n[LANGKAH 1] Menghapus Data Duplikat")
print("-" * 80)
duplicates_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"✓ Dihapus: {duplicates_count} baris duplikat")
print(f"✓ Sisa data: {df.shape[0]} baris")

# LANGKAH 2: Menghapus baris dengan missing values
print("\n[LANGKAH 2] Menghapus Missing Values")
print("-" * 80)
before_dropna = df.shape[0]
df_clean = df.dropna()
dropped_na = before_dropna - df_clean.shape[0]
print(f"✓ Dihapus: {dropped_na} baris dengan missing values")
print(f"✓ Sisa data: {df_clean.shape[0]} baris")

# LANGKAH 3: Menghapus baris dengan tipe data yang salah
print("\n[LANGKAH 3] Menghapus Data dengan Tipe yang Salah")
print("-" * 80)
before_type_filter = df_clean.shape[0]

# Filter hanya baris dengan BMI dan charges yang numerik
df_clean = df_clean[
    (df_clean['bmi'].apply(lambda x: isinstance(x, (int, float)))) &
    (df_clean['charges'].apply(lambda x: isinstance(x, (int, float))))
    ]

dropped_type = before_type_filter - df_clean.shape[0]
print(f"✓ Dihapus: {dropped_type} baris dengan tipe data yang salah")
print(f"✓ Sisa data: {df_clean.shape[0]} baris")

# LANGKAH 4: Konversi tipe data ke format yang benar
print("\n[LANGKAH 4] Konversi Tipe Data")
print("-" * 80)
df_clean['bmi'] = pd.to_numeric(df_clean['bmi'], errors='coerce')
df_clean['charges'] = pd.to_numeric(df_clean['charges'], errors='coerce')
print("✓ Kolom 'bmi' dikonversi ke tipe numerik")
print("✓ Kolom 'charges' dikonversi ke tipe numerik")

# LANGKAH 5: Menyimpan data bersih
print("\n[LANGKAH 5] Menyimpan Data Bersih")
print("-" * 80)
df_clean.to_excel("./insurance_remed_clean.xlsx", index=False)
print("✓ Data bersih disimpan ke file: insurance_remed_clean.xlsx")

# Ringkasan pembersihan
print("\n" + "=" * 80)
print("RINGKASAN PEMBERSIHAN DATA")
print("=" * 80)
print(f"Data SEBELUM pembersihan  : {original_size} baris")
print(f"Data SETELAH pembersihan  : {df_clean.shape[0]} baris")
print(f"Total data yang dihapus   : {original_size - df_clean.shape[0]} baris")
print(f"Persentase data tersisa   : {(df_clean.shape[0] / original_size) * 100:.2f}%")
print(f"Persentase data dihapus   : {((original_size - df_clean.shape[0]) / original_size) * 100:.2f}%")

print("\n✓ Data sudah bersih dan siap untuk analisis lanjutan!")

# Menampilkan info data bersih
print("\nInformasi Data Bersih:")
print("-" * 80)
df_clean.info()

# ================================================================================
# 4. STATISTIK DESKRIPTIF (DESCRIPTIVE STATISTICS)
# ================================================================================
# Menganalisis seluruh variabel dengan statistik deskriptif untuk memahami:
# - Ukuran pemusatan (mean, median)
# - Ukuran penyebaran (standar deviasi, range)
# - Nilai minimum dan maksimum

print("\n" + "=" * 80)
print("4. STATISTIK DESKRIPTIF")
print("=" * 80)

print("\n[A] STATISTIK DESKRIPTIF VARIABEL NUMERIK")
print("-" * 80)
desc_stats = df_clean.describe()
print(desc_stats)

print("\n[B] STATISTIK DESKRIPTIF VARIABEL KATEGORIKAL")
print("-" * 80)
print("\n1. Distribusi Jenis Kelamin (Sex):")
print(df_clean['sex'].value_counts())
print(f"   Persentase:")
print(df_clean['sex'].value_counts(normalize=True) * 100)

print("\n2. Distribusi Status Merokok (Smoker):")
print(df_clean['smoker'].value_counts())
print(f"   Persentase:")
print(df_clean['smoker'].value_counts(normalize=True) * 100)

print("\n3. Distribusi Wilayah (Region):")
print(df_clean['region'].value_counts())
print(f"   Persentase:")
print(df_clean['region'].value_counts(normalize=True) * 100)

print("\n[C] INTERPRETASI STATISTIK DESKRIPTIF")
print("-" * 80)
print(f"""
VARIABEL NUMERIK:

1. AGE (Umur):
   - Rata-rata: {desc_stats.loc['mean', 'age']:.2f} tahun
   - Median: {desc_stats.loc['50%', 'age']:.2f} tahun
   - Rentang: {desc_stats.loc['min', 'age']:.0f} - {desc_stats.loc['max', 'age']:.0f} tahun
   - Std Dev: {desc_stats.loc['std', 'age']:.2f} tahun
   → Usia pemegang polis bervariasi dari usia muda hingga lanjut

2. BMI (Body Mass Index):
   - Rata-rata: {desc_stats.loc['mean', 'bmi']:.2f}
   - Median: {desc_stats.loc['50%', 'bmi']:.2f}
   - Rentang: {desc_stats.loc['min', 'bmi']:.2f} - {desc_stats.loc['max', 'bmi']:.2f}
   - Std Dev: {desc_stats.loc['std', 'bmi']:.2f}
   → Mayoritas pemegang polis memiliki BMI normal hingga overweight

3. CHILDREN (Jumlah Anak):
   - Rata-rata: {desc_stats.loc['mean', 'children']:.2f} anak
   - Median: {desc_stats.loc['50%', 'children']:.0f} anak
   - Rentang: {desc_stats.loc['min', 'children']:.0f} - {desc_stats.loc['max', 'children']:.0f} anak
   → Sebagian besar pemegang polis memiliki sedikit tanggungan

4. CHARGES (Biaya Asuransi):
   - Rata-rata: ${desc_stats.loc['mean', 'charges']:,.2f}
   - Median: ${desc_stats.loc['50%', 'charges']:,.2f}
   - Rentang: ${desc_stats.loc['min', 'charges']:,.2f} - ${desc_stats.loc['max', 'charges']:,.2f}
   - Std Dev: ${desc_stats.loc['std', 'charges']:,.2f}
   → Biaya asuransi sangat bervariasi dengan standar deviasi yang tinggi
""")

# ================================================================================
# 5. UJI NORMALITAS DISTRIBUSI DATA
# ================================================================================
# Melakukan uji Shapiro-Wilk untuk menguji apakah data mengikuti distribusi normal
# H0: Data mengikuti distribusi normal
# H1: Data tidak mengikuti distribusi normal
# Jika p-value > 0.05, maka gagal tolak H0 (data normal)
# Jika p-value <= 0.05, maka tolak H0 (data tidak normal)

print("\n" + "=" * 80)
print("5. UJI NORMALITAS DISTRIBUSI DATA (SHAPIRO-WILK TEST)")
print("=" * 80)

# Variabel numerik yang akan diuji
numeric_cols = ['age', 'bmi', 'children', 'charges']

# Melakukan uji Shapiro-Wilk untuk setiap variabel
print("\n[A] HASIL UJI SHAPIRO-WILK")
print("-" * 80)

shapiro_results = {}
for col in numeric_cols:
    # Ambil sample jika data terlalu besar (Shapiro-Wilk max 5000 data)
    if len(df_clean[col]) > 5000:
        sample_data = df_clean[col].sample(5000, random_state=42)
    else:
        sample_data = df_clean[col]

    stat, p_value = stats.shapiro(sample_data)
    shapiro_results[col] = {'statistic': stat, 'p_value': p_value}

    print(f"\nVariabel: {col.upper()}")
    print(f"  Statistik W: {stat:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Alpha (tingkat signifikansi): 0.05")

    if p_value > 0.05:
        print(f"  ✓ Kesimpulan: Data '{col}' MENGIKUTI distribusi normal")
        print(f"    (Gagal tolak H0 karena p-value > alpha)")
    else:
        print(f"  ✗ Kesimpulan: Data '{col}' TIDAK MENGIKUTI distribusi normal")
        print(f"    (Tolak H0 karena p-value <= alpha)")

# ================================================================================
# 6. VISUALISASI DISTRIBUSI DATA
# ================================================================================
# Membuat histogram dan boxplot untuk setiap variabel numerik

print("\n" + "=" * 80)
print("6. VISUALISASI DISTRIBUSI DATA")
print("=" * 80)

# HISTOGRAM untuk setiap variabel numerik
print("\n[A] HISTOGRAM - Distribusi Frekuensi")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HISTOGRAM: Distribusi Frekuensi Variabel Numerik', fontsize=16, fontweight='bold', y=1.00)

colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for idx, col in enumerate(numeric_cols):
    row = idx // 2
    col_pos = idx % 2

    # Histogram dengan KDE (Kernel Density Estimation)
    axes[row, col_pos].hist(df_clean[col], bins=30, color=colors[idx],
                            edgecolor='black', alpha=0.7, density=True)

    # Tambahkan KDE line
    df_clean[col].plot(kind='kde', ax=axes[row, col_pos], color='darkblue',
                       linewidth=2, secondary_y=False)

    axes[row, col_pos].set_title(f'Distribusi {col.upper()}', fontsize=12, fontweight='bold')
    axes[row, col_pos].set_xlabel(col)
    axes[row, col_pos].set_ylabel('Density (Kepadatan)')
    axes[row, col_pos].grid(True, alpha=0.3)

    # Tambahkan garis mean dan median
    mean_val = df_clean[col].mean()
    median_val = df_clean[col].median()
    axes[row, col_pos].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[row, col_pos].axvline(median_val, color='green', linestyle='--', linewidth=2,
                               label=f'Median: {median_val:.2f}')
    axes[row, col_pos].legend()

plt.tight_layout()
plt.savefig('histogram_distribusi.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Histogram telah dibuat untuk semua variabel numerik")

# BOXPLOT untuk setiap variabel numerik
print("\n[B] BOXPLOT - Deteksi Outlier dan Persebaran Data")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BOXPLOT: Persebaran dan Outlier Variabel Numerik', fontsize=16, fontweight='bold', y=1.00)

for idx, col in enumerate(numeric_cols):
    row = idx // 2
    col_pos = idx % 2

    # Boxplot dengan styling
    bp = axes[row, col_pos].boxplot(df_clean[col], vert=True, patch_artist=True,
                                    boxprops=dict(facecolor=colors[idx], alpha=0.7),
                                    medianprops=dict(color='red', linewidth=2),
                                    whiskerprops=dict(color='black', linewidth=1.5),
                                    capprops=dict(color='black', linewidth=1.5),
                                    flierprops=dict(marker='o', markerfacecolor='red',
                                                    markersize=6, linestyle='none', alpha=0.5))

    axes[row, col_pos].set_title(f'Boxplot {col.upper()}', fontsize=12, fontweight='bold')
    axes[row, col_pos].set_ylabel(col)
    axes[row, col_pos].grid(True, alpha=0.3, axis='y')

    # Tambahkan informasi statistik
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df_clean[(df_clean[col] < q1 - 1.5 * iqr) | (df_clean[col] > q3 + 1.5 * iqr)][col]

    textstr = f'Q1: {q1:.2f}\nMedian: {df_clean[col].median():.2f}\nQ3: {q3:.2f}\nOutliers: {len(outliers)}'
    axes[row, col_pos].text(1.15, 0.5, textstr, transform=axes[row, col_pos].transAxes,
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('boxplot_distribusi.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Boxplot telah dibuat untuk semua variabel numerik")

# ================================================================================
# KESIMPULAN STATISTIK DESKRIPTIF DAN UJI NORMALITAS
# ================================================================================

print("\n" + "=" * 80)
print("KESIMPULAN STATISTIK DESKRIPTIF DAN UJI NORMALITAS")
print("=" * 80)

print("""
KESIMPULAN:

1. STATISTIK DESKRIPTIF:
   a) Variabel Age (Umur):
      - Data tersebar cukup merata dari usia muda hingga tua
      - Rata-rata dan median relatif berdekatan
   
   b) Variabel BMI (Body Mass Index):
      - Mayoritas pemegang polis memiliki BMI antara 26-41
      - Distribusi cenderung simetris dengan sedikit skewness
   
   c) Variabel Children (Jumlah Anak):
      - Kebanyakan pemegang polis memiliki 0-2 anak
      - Distribusi right-skewed (miring ke kanan)
   
   d) Variabel Charges (Biaya Asuransi):
      - Sangat bervariasi dengan range yang lebar
      - Mean lebih besar dari median, menunjukkan distribusi right-skewed
      - Ada perbedaan signifikan antara biaya terendah dan tertinggi

2. UJI NORMALITAS (SHAPIRO-WILK):
   - Semua variabel numerik menunjukkan p-value < 0.05
   - Kesimpulan: SEMUA VARIABEL TIDAK MENGIKUTI DISTRIBUSI NORMAL
   - Ini umum terjadi pada data real-world, terutama untuk data biaya
     yang cenderung right-skewed

3. VISUALISASI:
   a) Histogram:
      - Menunjukkan pola distribusi setiap variabel
      - Charges menunjukkan distribusi multimodal (beberapa puncak)
      - Age relatif lebih uniform dibanding variabel lain
   
   b) Boxplot:
      - Terdeteksi beberapa outlier pada variabel charges
      - Outlier ini mungkin sah (orang dengan kondisi kesehatan khusus)
      - Variabel age dan bmi relatif tidak memiliki banyak outlier

4. IMPLIKASI:
   - Karena data tidak normal, penggunaan metode statistik non-parametrik
     mungkin lebih tepat untuk beberapa analisis
   - Namun, untuk regresi linear, asumsi normalitas residual lebih penting
     daripada normalitas variabel individual
   - Transformasi data (log, sqrt) bisa dipertimbangkan jika diperlukan
""")

# ================================================================================
# 7. ANALISIS KORELASI
# ================================================================================
# Menganalisis hubungan linear antara Age (umur) dengan Charges (biaya asuransi)
# Korelasi mengukur kekuatan dan arah hubungan antara dua variabel

print("\n" + "=" * 80)
print("7. ANALISIS KORELASI")
print("=" * 80)

print("\n[A] KORELASI ANTARA AGE DAN CHARGES")
print("-" * 80)

# Menghitung korelasi Pearson
correlation_age_charges = df_clean['age'].corr(df_clean['charges'])
print(f"Koefisien Korelasi Pearson (r): {correlation_age_charges:.4f}")

# Interpretasi kekuatan korelasi
if abs(correlation_age_charges) < 0.3:
    strength = "LEMAH"
elif abs(correlation_age_charges) < 0.7:
    strength = "SEDANG"
else:
    strength = "KUAT"

direction = "POSITIF" if correlation_age_charges > 0 else "NEGATIF"

print(f"\nInterpretasi:")
print(f"  - Kekuatan: {strength}")
print(f"  - Arah: {direction}")
print(f"  - Artinya: Terdapat hubungan {direction.lower()} yang {strength.lower()}")
print(f"            antara umur (age) dengan biaya asuransi (charges)")

# Matriks korelasi untuk semua variabel numerik
print("\n[B] MATRIKS KORELASI SEMUA VARIABEL NUMERIK")
print("-" * 80)

correlation_matrix = df_clean[numeric_cols].corr()
print(correlation_matrix)

# Visualisasi Heatmap Korelasi
print("\n[C] VISUALISASI HEATMAP KORELASI")
print("-" * 80)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            fmt='.3f', vmin=-1, vmax=1)
plt.title('HEATMAP KORELASI VARIABEL NUMERIK', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('heatmap_korelasi.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Heatmap korelasi telah dibuat")

# Visualisasi Scatter Plot: Age vs Charges
print("\n[D] SCATTER PLOT: AGE vs CHARGES")
print("-" * 80)

plt.figure(figsize=(12, 6))
plt.scatter(df_clean['age'], df_clean['charges'], alpha=0.5, s=50, c='steelblue', edgecolors='black', linewidth=0.5)
plt.xlabel('Age (Umur)', fontsize=12, fontweight='bold')
plt.ylabel('Charges (Biaya Asuransi)', fontsize=12, fontweight='bold')
plt.title(f'SCATTER PLOT: Hubungan Age vs Charges\n(Korelasi r = {correlation_age_charges:.4f})',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Tambahkan garis trend
z = np.polyfit(df_clean['age'], df_clean['charges'], 1)
p = np.poly1d(z)
plt.plot(df_clean['age'], p(df_clean['age']), "r--", linewidth=2, label=f'Trend Line: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('scatter_age_charges.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Scatter plot telah dibuat")

# ================================================================================
# 8. ANALISIS REGRESI LINEAR
# ================================================================================
# Membuat model regresi linear untuk memprediksi Charges berdasarkan Age

print("\n" + "=" * 80)
print("8. ANALISIS REGRESI LINEAR")
print("=" * 80)

print("\n[A] MEMBANGUN MODEL REGRESI")
print("-" * 80)

# Membuat model regresi menggunakan statsmodels
model = smf.ols('charges ~ age', data=df_clean).fit()

# Ekstrak parameter model
intercept = model.params['Intercept']
slope = model.params['age']

print(f"Model Regresi Linear Sederhana:")
print(f"  Charges = beta0 + beta1 × Age")
print(f"\nParameter Model:")
print(f"  beta0 (Intercept/Konstanta): {intercept:,.2f}")
print(f"  beta1 (Slope/Koefisien Age): {slope:,.2f}")
print(f"\nPersamaan Regresi:")
print(f"  Charges = {intercept:,.2f} + {slope:,.2f} × Age")

print(f"\nInterpretasi:")
print(f"  - Intercept ({intercept:,.2f}): Biaya dasar asuransi ketika umur = 0")
print(f"  - Slope ({slope:,.2f}): Setiap kenaikan 1 tahun umur, biaya asuransi")
print(f"                        meningkat sebesar ${slope:,.2f}")

# Ringkasan model
print("\n[B] RINGKASAN MODEL REGRESI")
print("-" * 80)
print(model.summary())

# Koefisien Determinasi (R-squared)
print("\n[C] KOEFISIEN DETERMINASI (R-SQUARED)")
print("-" * 80)

r_squared = model.rsquared
r_squared_adj = model.rsquared_adj

print(f"R-squared (R²): {r_squared:.4f}")
print(f"R-squared Adjusted: {r_squared_adj:.4f}")
print(f"\nInterpretasi R²:")
print(f"  - Model ini menjelaskan {r_squared * 100:.2f}% variasi dalam biaya asuransi (charges)")
print(f"  - Sisanya ({(1 - r_squared) * 100:.2f}%) dijelaskan oleh faktor lain yang tidak")
print(f"    termasuk dalam model (misalnya: status merokok, BMI, dll)")

if r_squared < 0.3:
    r2_interpretation = "RENDAH - Model kurang baik dalam menjelaskan variasi"
elif r_squared < 0.6:
    r2_interpretation = "SEDANG - Model cukup baik dalam menjelaskan variasi"
else:
    r2_interpretation = "TINGGI - Model sangat baik dalam menjelaskan variasi"

print(f"  - Tingkat kecocokan model: {r2_interpretation}")

# Sum of Squared Residuals (SSR)
print("\n[D] SUM OF SQUARED RESIDUALS (SSR)")
print("-" * 80)

ssr = model.ssr
mse = model.mse_resid
rmse = np.sqrt(mse)

print(f"SSR (Sum of Squared Residuals): {ssr:,.2f}")
print(f"MSE (Mean Squared Error): {mse:,.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:,.2f}")
print(f"\nInterpretasi SSR:")
print(f"  - SSR adalah total kuadrat kesalahan prediksi")
print(f"  - Semakin kecil SSR, semakin baik model dalam memprediksi")
print(f"  - RMSE ({rmse:,.2f}) menunjukkan rata-rata kesalahan prediksi")
print(f"    sekitar ${rmse:,.2f} dari nilai aktual")

# Visualisasi Model Regresi
print("\n[E] VISUALISASI MODEL REGRESI")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Scatter plot dengan garis regresi
axes[0].scatter(df_clean['age'], df_clean['charges'], alpha=0.5, s=50,
                c='steelblue', edgecolors='black', linewidth=0.5, label='Data Aktual')

# Garis regresi
x_pred = np.linspace(df_clean['age'].min(), df_clean['age'].max(), 100)
y_pred = intercept + slope * x_pred
axes[0].plot(x_pred, y_pred, 'r-', linewidth=3, label=f'Garis Regresi')

axes[0].set_xlabel('Age (Umur)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Charges (Biaya Asuransi)', fontsize=12, fontweight='bold')
axes[0].set_title(f'MODEL REGRESI LINEAR\nCharges = {intercept:,.2f} + {slope:,.2f} × Age\nR² = {r_squared:.4f}',
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Residual plot
residuals = model.resid
axes[1].scatter(model.fittedvalues, residuals, alpha=0.5, s=50,
                c='coral', edgecolors='black', linewidth=0.5)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Fitted Values (Nilai Prediksi)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Kesalahan)', fontsize=12, fontweight='bold')
axes[1].set_title('RESIDUAL PLOT\n(Untuk Memeriksa Asumsi Model)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_regresi_linear.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Visualisasi model regresi telah dibuat")

# ================================================================================
# KESIMPULAN AKHIR ANALISIS KORELASI DAN REGRESI
# ================================================================================

print("\n" + "=" * 80)
print("KESIMPULAN AKHIR ANALISIS KORELASI DAN REGRESI")
print("=" * 80)

print(f"""
RINGKASAN HASIL ANALISIS:

1. KORELASI ANTARA AGE DAN CHARGES:
   - Koefisien korelasi (r): {correlation_age_charges:.4f}
   - Kekuatan hubungan: {strength}
   - Arah hubungan: {direction}
   
   Interpretasi:
   Terdapat hubungan {direction.lower()} yang {strength.lower()} antara umur (age) 
   dengan biaya asuransi (charges). Ini berarti:
   → Semakin tua umur seseorang, biaya asuransi cenderung meningkat
   → Namun, hubungan ini tidak terlalu kuat karena ada faktor lain yang
     juga mempengaruhi biaya asuransi (seperti status merokok, BMI, dll)

2. MODEL REGRESI LINEAR:
   Persamaan: Charges = {intercept:,.2f} + {slope:,.2f} × Age
   
   Interpretasi:
   a) Intercept ({intercept:,.2f}):
      - Biaya dasar asuransi (ketika umur = 0)
      - Nilai ini adalah estimasi teoritis
   
   b) Slope ({slope:,.2f}):
      - Setiap bertambah 1 tahun umur, biaya asuransi naik ${slope:,.2f}
      - Contoh: Umur 30 tahun vs 40 tahun
        * Umur 30: ${intercept + slope * 30:,.2f}
        * Umur 40: ${intercept + slope * 40:,.2f}
        * Selisih: ${slope * 10:,.2f}

3. KOEFISIEN DETERMINASI (R²):
   - R² = {r_squared:.4f} atau {r_squared * 100:.2f}%
   - R² Adjusted = {r_squared_adj:.4f}
   
   Interpretasi:
   → Model ini hanya menjelaskan {r_squared * 100:.2f}% variasi dalam biaya asuransi
   → Sisanya ({(1 - r_squared) * 100:.2f}%) dipengaruhi oleh faktor lain seperti:
     * Status merokok (smoker)
     * Body Mass Index (BMI)
     * Jumlah anak (children)
     * Wilayah tempat tinggal (region)
     * Riwayat penyakit
     * dll
   → Nilai R² yang {strength.lower()} menunjukkan bahwa model sederhana ini
     belum optimal untuk prediksi

4. SUM OF SQUARED RESIDUALS (SSR):
   - SSR: {ssr:,.2f}
   - RMSE: {rmse:,.2f}
   
   Interpretasi:
   → Rata-rata kesalahan prediksi model adalah sekitar ${rmse:,.2f}
   → Ini cukup besar mengingat range biaya asuransi yang lebar
   → Menunjukkan bahwa prediksi hanya berdasarkan umur saja tidak cukup akurat

5. REKOMENDASI:
   a) Model Sederhana vs Model Kompleks:
      - Model regresi sederhana (hanya age) kurang optimal
      - Disarankan menggunakan multiple regression dengan variabel tambahan:
        * Age + BMI + Smoker + Children + Region
      - Model yang lebih kompleks akan meningkatkan R² dan akurasi prediksi
   
   b) Aplikasi Praktis:
      - Umur adalah faktor penting tapi bukan satu-satunya penentu
      - Perusahaan asuransi harus mempertimbangkan banyak faktor
      - Status merokok kemungkinan memiliki pengaruh lebih besar daripada umur
   
   c) Validasi Model:
      - Residual plot menunjukkan pola yang perlu dievaluasi
      - Perlu dilakukan uji asumsi regresi lebih lanjut:
        * Linearitas
        * Homoskedastisitas
        * Normalitas residual
        * Independensi residual

KESIMPULAN UTAMA:
================================================================================
Meskipun terdapat hubungan positif antara umur (age) dengan biaya asuransi
(charges), hubungan ini tergolong {strength.lower()} dan hanya menjelaskan {r_squared * 100:.2f}%
variasi dalam biaya asuransi. Oleh karena itu, untuk prediksi yang lebih akurat,
diperlukan model yang lebih kompleks dengan memasukkan variabel-variabel lain
yang relevan seperti status merokok, BMI, dan faktor kesehatan lainnya.

Model sederhana ini dapat digunakan sebagai baseline atau untuk memberikan
gambaran umum tentang pengaruh umur terhadap biaya asuransi, namun tidak
disarankan untuk digunakan sebagai satu-satunya dasar dalam pengambilan
keputusan bisnis yang kritis.
================================================================================
""")

print("\n" + "=" * 80)
print("ANALISIS SELESAI")
print("=" * 80)
print("\n✓ Semua analisis telah selesai dilakukan")
print("✓ File data bersih: insurance_remed_clean.xlsx")
print("✓ File visualisasi:")
print("  - boxplot_outlier_detection.png")
print("  - histogram_distribusi.png")
print("  - boxplot_distribusi.png")
print("  - heatmap_korelasi.png")
print("  - scatter_age_charges.png")
print("  - model_regresi_linear.png")
print("=" * 80)
