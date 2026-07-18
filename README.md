# UAS Kecerdasan Buatan — Prediksi Status Gizi Balita Menggunakan KNN & Random Forest

Repository ini berisi tugas UAS mata kuliah Kecerdasan Buatan: klasifikasi status gizi balita
(`normal`, `stunted`, `severely stunted`, `tinggi`) berdasarkan umur, jenis kelamin, dan tinggi
badan, menggunakan **K-Nearest Neighbor (KNN)** sebagai model utama dan **Random Forest**
sebagai model pembanding.

## Anggota Kelompok
- Nama 1 — NIM
- Nama 2 — NIM

## Struktur Repository

```
UAS-KecerdasanBuatan/
├── README.md              # Dokumen ini
├── Laporan_uas.md          # Laporan lengkap (business understanding s/d referensi)
├── uas_model.ipynb         # Notebook: EDA, preprocessing, modeling, evaluasi
├── data/
│   └── data_balita.csv     # Dataset mentah
└── Jurnal/                 # Kumpulan jurnal referensi (PDF)
```

## Langkah Pengerjaan

1. **Data Collection** — Dataset status gizi balita (`data_balita.csv`) diletakkan di folder
   `data/`. Notebook membaca file ini secara otomatis (atau melalui dialog upload jika
   dijalankan di Google Colab).
2. **Exploratory Data Analysis (EDA)** — Memeriksa struktur data, missing value, duplikasi,
   distribusi kelas target, distribusi umur/tinggi badan, serta pola umur–tinggi badan per
   kelas status gizi.
3. **Data Preparation** — Membersihkan kategori (strip/lowercase), mengonversi kolom numerik,
   memvalidasi rentang umur (0–60 bulan) dan tinggi badan (30–150 cm), membuang baris dengan
   label tidak valid dan data duplikat identik, lalu melakukan `train_test_split` (80:20,
   stratified) diikuti `StandardScaler` untuk fitur numerik dan `OneHotEncoder` untuk fitur
   kategorik (jenis kelamin) melalui `ColumnTransformer`.
4. **Modeling** — Dua algoritma diterapkan pada pipeline preprocessing yang sama:
   - **KNN**: nilai K dipilih dari kandidat `[3, 5, 7, 9, 11, 15, 21]` berdasarkan F1-Score
     macro pada subset validasi, lalu model final dilatih ulang pada seluruh data latih.
   - **Random Forest**: `n_estimators=300`, `class_weight='balanced'`, dilatih pada data latih
     yang sama sebagai pembanding.
5. **Evaluation** — Kedua model dievaluasi pada test set yang sama menggunakan Accuracy,
   Precision, Recall, F1-Score (macro & weighted), serta confusion matrix.
6. **Model Saving & Inference** — Pipeline KNN (preprocessing + model) disimpan sebagai file
   `joblib` beserta metadata, lalu diuji dengan beberapa contoh data baru.
7. **Pelaporan** — Seluruh tahap, hasil, dan analisis dituliskan secara naratif di
   `Laporan_uas.md`, lengkap dengan referensi ilmiah.

## Cara Menjalankan

1. Buka `uas_model.ipynb` di Google Colab atau Jupyter lokal.
2. Pastikan `data_balita.csv` tersedia di folder `data/` (atau upload manual saat diminta).
3. Jalankan seluruh cell secara berurutan dari atas ke bawah (`Run All`).
4. Model dan metadata hasil training akan tersimpan otomatis di direktori output.

## Ringkasan Hasil

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|
| KNN (K=3) | 0.9916 | 0.9882 | 0.9883 | 0.9882 |
| Random Forest | *(isi setelah menjalankan notebook)* | | | |

![alt text](image.png)

Detail lengkap ada di [`Laporan_uas.md`](./Laporan_uas.md).
