# Laporan UAS Kecerdasan Buatan

## 1. Judul Proyek

**Prediksi Status Gizi Balita Menggunakan Algoritma K-Nearest Neighbor (KNN) dan Random Forest**

**Nama Kelompok:**
- Moch Zidane Bahtiar  — 2406073
- M Gilan Algina — 2406101

**Domain Proyek (Latar Belakang)**

Status gizi balita merupakan indikator penting kesehatan anak dan kualitas sumber daya manusia
di masa depan. Gangguan gizi kronis pada balita, khususnya *stunting* (tubuh pendek akibat
kekurangan gizi kronis), masih menjadi tantangan besar di Indonesia karena berdampak pada
pertumbuhan fisik, perkembangan kognitif, dan produktivitas anak di masa depan (Khotimah, 2022).
Deteksi status gizi secara manual oleh tenaga medis melalui pengukuran antropometri memerlukan
waktu dan rentan terhadap ketidaktelitian, sehingga pendekatan *machine learning* dapat menjadi
alat bantu skrining awal yang cepat dan konsisten (Lonang & Normawati, 2022). Proyek ini
menerapkan algoritma **K-Nearest Neighbor (KNN)** sebagai model utama, dibandingkan dengan
**Random Forest**, untuk mengklasifikasikan status gizi balita ke dalam empat kelas: `normal`,
`stunted`, `severely stunted`, dan `tinggi`, berdasarkan tiga fitur: umur (bulan), jenis
kelamin, dan tinggi badan (cm).

---

## 2. Business Understanding

**Permasalahan dunia nyata dan literatur review**

Pemeriksaan status gizi balita secara konvensional bergantung pada kunjungan ke posyandu/tenaga
kesehatan dan interpretasi manual terhadap standar antropometri, yang tidak selalu tersedia
secara real-time di semua wilayah. Beberapa penelitian terdahulu telah menerapkan algoritma
klasifikasi seperti KNN, Naive Bayes, SVM, dan Random Forest untuk memprediksi status gizi atau
status stunting balita dengan akurasi yang bervariasi (81%–99%) tergantung kualitas data dan
algoritma yang digunakan (Lonang & Normawati, 2022; Panigoro, 2024; Ariyadi et al., 2024).
Random Forest secara konsisten dilaporkan memberikan akurasi yang stabil dan tinggi pada kasus
klasifikasi stunting dibandingkan algoritma pembanding seperti SVM (Panigoro, 2024).

**Tujuan proyek**
- Membangun model klasifikasi multikelas status gizi balita menggunakan KNN.
- Membandingkan performa KNN dengan Random Forest pada data dan skema evaluasi yang sama.
- Menghasilkan pipeline model yang dapat disimpan dan digunakan untuk inference pada data baru.

**Siapa user/pengguna sistem**

Tenaga kesehatan masyarakat (kader posyandu, bidan, ahli gizi) serta pengembang aplikasi
kesehatan yang membutuhkan sistem bantu skrining awal status gizi balita berbasis data
antropometri sederhana (umur, jenis kelamin, tinggi badan).

**Solusi dan manfaat implementasi AI**

Model klasifikasi berbasis AI memungkinkan skrining status gizi dilakukan secara instan dari
tiga input sederhana, tanpa memerlukan tabel rujukan antropometri manual. Model dapat
diintegrasikan ke aplikasi web/mobile (Flask/FastAPI) sebagai alat bantu deteksi dini, bukan
pengganti pemeriksaan tenaga kesehatan profesional.

---

## 3. Data Understanding

**Sumber data**: Dataset `data_balita.csv` — data tabular status gizi balita (simulasi/open
dataset), diunggah melalui Google Colab.

**Deskripsi fitur**

| Fitur                  | Tipe                 | Keterangan                                         |
|:-----------------------|:---------------------|:----------------------------------------------------|
| Umur (bulan)           | Numerik (int)         | Rentang valid 0–60 bulan                           |
| Jenis Kelamin          | Kategorik             | `laki-laki` / `perempuan`                          |
| Tinggi Badan (cm)      | Numerik (float)       | Rentang valid 30–150 cm                            |
| Status Gizi (target)   | Kategorik (4 kelas)   | `normal`, `stunted`, `severely stunted`, `tinggi`   |

**Ukuran dan format data**

Data mentah berjumlah **120.999 baris × 4 kolom**, format CSV, tanpa missing value pada kolom
asli. Setelah proses pembersihan (lihat bagian Data Preparation), tersisa **39.425 baris** data
valid dan unik yang digunakan untuk pemodelan.

**Tipe data dan target klasifikasi**

Target `Status Gizi` merupakan variabel kategorik dengan 4 kelas (klasifikasi multikelas),
dengan distribusi awal sebelum pembersihan: `normal` 56,0%, `severely stunted` 16,4%, `tinggi`
16,2%, `stunted` 11,4% — menunjukkan data tidak seimbang (*imbalanced*), dengan kelas `normal`
mendominasi.

---

## 4. Exploratory Data Analysis (EDA)

**Visualisasi distribusi data**
- Bar chart distribusi `Status Gizi` menunjukkan dominasi kelas `normal`.
- Bar chart distribusi `Jenis Kelamin` menunjukkan proporsi yang seimbang antara laki-laki dan
  perempuan.
- Histogram `Umur (bulan)` menunjukkan sebaran umur relatif merata dari 0 hingga 60 bulan.
- Histogram `Tinggi Badan (cm)` menunjukkan distribusi mendekati normal dengan rata-rata
  ±88,7 cm.

**Analisis korelasi antar fitur**

Scatter plot umur vs tinggi badan yang diwarnai per kelas status gizi menunjukkan pola yang
jelas: balita dengan tinggi badan jauh di bawah kurva pertumbuhan normal pada umur tertentu
cenderung diklasifikasikan `severely stunted`/`stunted`, sedangkan yang berada di atas
cenderung `tinggi`. Pola ini konsisten dengan prinsip antropometri gizi (tinggi badan relatif
terhadap umur) yang menjadi dasar klasifikasi stunting (Titimeidara & Hadikurniawati, 2021).

**Deteksi data tidak seimbang (imbalanced classes)**

Setelah data preparation, distribusi kelas: `normal` 21.514 (54,6%), `tinggi` 6.974 (17,7%),
`severely stunted` 6.520 (16,5%), `stunted` 4.417 (11,2%). Data tetap tidak seimbang, sehingga
metrik evaluasi menggunakan **rata-rata macro** (bukan hanya accuracy) agar kelas minoritas
tetap terukur secara adil, dan pada Random Forest digunakan parameter `class_weight='balanced'`
untuk mengurangi bias terhadap kelas mayoritas.

**Insight awal**

Fitur umur dan tinggi badan secara bersama memiliki hubungan non-linear yang kuat dengan label
status gizi, sedangkan jenis kelamin berkontribusi sebagai faktor penyesuaian standar
pertumbuhan. Hal ini mengindikasikan algoritma berbasis jarak (KNN) dan berbasis pohon
keputusan (Random Forest) sama-sama berpotensi menangkap pola tersebut dengan baik.

---

## 5. Data Preparation

**Pembersihan data**
- Normalisasi teks kategorik: `strip()`, `lower()`, penghapusan spasi ganda pada kolom
  `Jenis Kelamin` dan `Status Gizi`.
- Konversi kolom numerik (`Umur (bulan)`, `Tinggi Badan (cm)`) ke tipe numerik, nilai yang gagal
  dikonversi menjadi `NaN`.
- Nilai umur di luar rentang 0–60 bulan dan tinggi badan di luar 30–150 cm diset menjadi `NaN`
  (dianggap tidak valid secara antropometri).
- Baris dengan label target di luar 4 kelas valid dibuang.
- Data duplikat identik dibuang: **81.574 baris duplikat** dihapus dari 120.999 baris awal,
  menyisakan 39.425 baris unik dan valid.

**Encoding data kategorik**

Kolom `Jenis Kelamin` di-*encode* menggunakan `OneHotEncoder` (`handle_unknown='ignore'`) di
dalam `ColumnTransformer`, sehingga menghasilkan kolom biner terpisah untuk setiap kategori.

**Normalisasi/standardisasi data numerik**

Kolom `Umur (bulan)` dan `Tinggi Badan (cm)` diimputasi dengan nilai median (`SimpleImputer`)
lalu distandardisasi menggunakan `StandardScaler`, penting terutama untuk KNN karena algoritma
ini sensitif terhadap skala fitur (jarak Euclidean).

**Split data (train-test)**

Data dibagi 80:20 menggunakan `train_test_split` dengan `stratify=y` agar proporsi kelas tetap
terjaga di kedua subset:
- **Data latih**: 31.540 baris
- **Data uji**: 7.885 baris

---

## 6. Modeling

**Pemilihan algoritma**

Digunakan dua algoritma klasifikasi:
1. **K-Nearest Neighbor (KNN)** — model utama.
2. **Random Forest** — model pembanding.

**Alasan pemilihan**

- **KNN** dipilih karena sederhana, non-parametrik, dan efektif untuk data dengan pola
  kedekatan geometris yang jelas seperti hubungan umur–tinggi badan terhadap status gizi
  (Lonang & Normawati, 2022).
- **Random Forest** dipilih sebagai pembanding karena merupakan metode *ensemble* berbasis
  pohon keputusan yang terbukti memberikan akurasi tinggi dan stabil pada kasus klasifikasi
  stunting/status gizi balita di berbagai penelitian (Panigoro, 2024; Ariyadi et al., 2024;
  Handayani et al., 2024), tidak sensitif terhadap skala fitur, tahan terhadap *overfitting*
  melalui agregasi banyak pohon, serta menyediakan `feature_importances_` untuk interpretasi —
  sesuatu yang tidak dimiliki KNN secara langsung. Kedua model menggunakan pipeline
  preprocessing (`ColumnTransformer`) yang identik agar perbandingan adil.

**Implementasi model**

- KNN: nilai K terbaik dipilih dari kandidat `[3, 5, 7, 9, 11, 15, 21]` menggunakan skema
  tuning (fit/validation split dari subset data latih), dievaluasi dengan F1-Score macro. Nilai
  **K=3** terpilih sebagai yang terbaik, lalu model final dilatih ulang pada seluruh data latih
  (31.540 baris; waktu fit ±0,12 detik).
- Random Forest: `n_estimators=300`, `min_samples_leaf=2`, `class_weight='balanced'`,
  `random_state=42`, dilatih pada data latih yang sama.

```python
# KNN
knn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='uniform',
                                  algorithm='kd_tree', metric='minkowski', p=2))
])

# Random Forest (pembanding)
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight='balanced', random_state=42))
])
```

**Perbandingan model**

| Metrik                  | KNN (K=3) | Random Forest | Selisih (RF − KNN) |
|:------------------------|:---------:|:--------------:|:-------------------:|
| Accuracy                |  0,9912   |     0,9909     |       −0,0004        |
| Precision (Macro)       |  0,9883   |     0,9864     |       −0,0019        |
| Recall (Macro)          |  0,9874   |     0,9874     |       −0,0001        |
| F1-Score (Macro)        |  0,9879   |     0,9869     |       −0,0010        |
| Waktu Fit (detik)       |   0,12    |      6,92      |          —           |
| Waktu Prediksi (detik)  |   0,06    |      0,29      |          —           |

> Seluruh nilai pada tabel di atas dihasilkan dari eksekusi penuh `uas_model.ipynb` (KNN dan
> Random Forest dilatih pada `X_train`/`y_train` yang identik dan diuji pada `X_test`/`y_test`
> yang sama, `random_state=42`), sehingga perbandingan kedua model bersifat adil (*fair
> comparison*). Random Forest membutuhkan waktu *fit* jauh lebih lama (±6,92 detik) dibanding KNN
> (±0,12 detik) karena harus membangun 300 pohon keputusan, namun keduanya sama-sama cepat pada
> tahap prediksi.

**Visualisasi model**

Grafik performa terhadap nilai K (`Accuracy` dan `F1-Score Macro` untuk setiap kandidat K) serta
bar chart perbandingan metrik KNN vs Random Forest tersedia pada notebook `uas_model.ipynb`.

---

## 7. Evaluation

**Confusion matrix**

Confusion matrix (absolut dan ternormalisasi) untuk kedua model divisualisasikan pada notebook.
Dari 7.885 data uji:
- **KNN (K=3)** memprediksi **7.816 benar** dan **69 salah**.
- **Random Forest** memprediksi **7.813 benar** dan **72 salah**.

**Metrik evaluasi (KNN, model utama)**

| Kelas                | Precision | Recall | F1-Score | Support |
|:----------------------|:---------:|:------:|:--------:|--------:|
| normal                |  0,9944   | 0,9951 |  0,9948  |   4.303 |
| severely stunted      |  0,9900   | 0,9916 |  0,9908  |   1.304 |
| stunted               |  0,9772   | 0,9717 |  0,9744  |     883 |
| tinggi                |  0,9914   | 0,9914 |  0,9914  |   1.395 |
| **Accuracy**          | **0,9912** | **0,9912** | **0,9912** |   7.885 |
| **Macro avg**         |  0,9883   | 0,9874 |  0,9879  |   7.885 |
| **Weighted avg**      |  0,9912   | 0,9912 |  0,9912  |   7.885 |

> *Catatan: pada klasifikasi multi-kelas single-label (setiap balita hanya memiliki satu label
> aktual), precision dan recall rata-rata mikro (micro-average) secara matematis selalu sama
> dengan accuracy, sehingga baris Accuracy di atas diisi dengan nilai yang identik pada ketiga
> kolom.*

**Metrik evaluasi (Random Forest, model pembanding)**

| Kelas                | Precision | Recall | F1-Score | Support |
|:----------------------|:---------:|:------:|:--------:|--------:|
| normal                |  0,9958   | 0,9942 |  0,9950  |   4.303 |
| severely stunted      |  0,9870   | 0,9908 |  0,9889  |   1.304 |
| stunted               |  0,9706   | 0,9717 |  0,9711  |     883 |
| tinggi                |  0,9921   | 0,9928 |  0,9925  |   1.395 |
| **Accuracy**          | **0,9909** | **0,9909** | **0,9909** |   7.885 |
| **Macro avg**         |  0,9864   | 0,9874 |  0,9869  |   7.885 |
| **Weighted avg**      |  0,9909   | 0,9909 |  0,9909  |   7.885 |

**Penjelasan kinerja model**

Kedua model memberikan performa yang sangat tinggi dan hampir setara pada data uji. **KNN
(K=3)** mencatat accuracy 99,12% dan F1-Score macro 98,79%, sementara **Random Forest** mencatat
accuracy 99,09% dan F1-Score macro 98,69% — selisih keduanya sangat tipis (F1-Macro berbeda
0,0010, atau sekitar 0,1 poin persentase). Pada kedua model, kelas `stunted` konsisten menjadi
kelas dengan F1-Score terendah (KNN 97,44%; Random Forest 97,11%) karena jumlah datanya paling
sedikit (support 883 dari 7.885, atau 11,2%) — sejalan dengan kondisi *imbalanced class* yang
teridentifikasi saat EDA. Menariknya, Random Forest justru sedikit lebih unggul pada kelas
`severely stunted` dan `tinggi`, sementara KNN lebih unggul pada kelas `normal` dan `stunted`,
menunjukkan kedua algoritma memiliki karakteristik kesalahan klasifikasi yang sedikit berbeda
meski keduanya memakai pipeline preprocessing yang identik.

**Model terbaik ditentukan berdasarkan F1-Score macro tertinggi** pada data uji: **KNN (K=3)**
terpilih sebagai model utama karena F1-Score macro-nya (0,9879) sedikit lebih tinggi dibanding
Random Forest (0,9869), sekaligus memiliki waktu *fit* jauh lebih singkat (0,12 detik vs 6,92
detik). Hasil ini konsisten dengan temuan Lonang & Normawati (2022) bahwa KNN efektif pada data
dengan pola kedekatan geometris yang jelas seperti hubungan umur–tinggi badan terhadap status
gizi, meskipun literatur lain (Panigoro, 2024) melaporkan Random Forest umumnya lebih stabil
pada kasus klasifikasi stunting secara umum — perbedaan ini kemungkinan dipengaruhi oleh
karakteristik dataset (jumlah fitur yang sedikit dan hubungan fitur-target yang sangat
terstruktur) yang menguntungkan pendekatan berbasis jarak.

---

## 8. Kesimpulan dan Rekomendasi

**Ringkasan hasil modeling dan evaluasi**

Model **KNN (K=3)** berhasil mengklasifikasikan status gizi balita dengan accuracy **99,12%**
dan F1-Score macro **98,79%** pada data uji, menunjukkan performa yang sangat baik untuk kasus
klasifikasi multikelas ini. Sebagai pembanding, **Random Forest** (dengan pipeline preprocessing
identik) mencatat accuracy **99,09%** dan F1-Score macro **98,69%** — hanya berselisih tipis
0,10 poin dari KNN. Berdasarkan F1-Score macro tertinggi, **KNN (K=3) ditetapkan sebagai model
terbaik**, unggul tipis dari Random Forest sekaligus jauh lebih efisien dari sisi waktu
pelatihan (0,12 detik vs 6,92 detik).

**Apakah tujuan proyek tercapai?**

Ya. Model klasifikasi status gizi balita berhasil dibangun, dievaluasi, dan disimpan dalam
bentuk pipeline siap pakai (`joblib`) yang dapat digunakan untuk inference data baru, serta
telah dibandingkan secara adil dengan Random Forest sebagai model kedua sesuai kebutuhan tugas.

**Kelebihan dan keterbatasan model**

- *Kelebihan*: Akurasi tinggi, pipeline preprocessing terintegrasi (tidak perlu normalisasi
  manual saat deployment), tersedia estimasi *confidence* per prediksi.
- *Keterbatasan*: KNN memerlukan penyimpanan seluruh data latih dan menjadi lambat pada dataset
  sangat besar saat inference; hanya menggunakan 3 fitur sederhana sehingga belum
  mempertimbangkan faktor lain (berat badan, riwayat gizi ibu, dsb.) yang relevan secara medis;
  data tetap tidak seimbang sehingga kelas minoritas (`stunted`) memiliki performa relatif lebih
  rendah dibanding kelas lain.

**Rekomendasi perbaikan**

- Menambahkan fitur lain yang relevan secara medis (mis. berat badan, indikator gizi ibu saat
  hamil) untuk meningkatkan akurasi terutama pada kelas minoritas.
- Menerapkan teknik penyeimbangan data seperti SMOTE untuk mengurangi bias terhadap kelas
  mayoritas.
- Mengeksplorasi algoritma lain (SVM, Gradient Boosting/XGBoost) sebagai pembanding tambahan.
- Melakukan validasi dengan data dari sumber berbeda (data riil posyandu) untuk menguji
  generalisasi model di luar dataset simulasi.

---

## 9. Referensi

Ariyadi, M. R. A., Lestanti, S., & Kirom, S. (2024). Klasifikasi balita stunting menggunakan
Random Forest Classifier di Kabupaten Blitar. *JATI (Jurnal Mahasiswa Teknik Informatika)*,
7(6), 3846–3851. https://doi.org/10.36040/jati.v7i6.7822

Arisandi, R. R. R., Warsito, B., & Hakim, A. R. (2022). Aplikasi Naïve Bayes Classifier (NBC)
pada klasifikasi status gizi balita stunting dengan pengujian K-Fold Cross Validation. *Jurnal
Gaussian*, 11(1), 130–139. https://doi.org/10.14710/j.gauss.v11i1.33991

Handayani, P., Fauzan, A. C., & Harliana, H. (2024). Machine learning klasifikasi status gizi
balita menggunakan algoritma Random Forest. *KLIK: Kajian Ilmiah Informatika dan Komputer*,
4(6), 3064–3072.

Lonang, S., & Normawati, D. (2022). Klasifikasi status stunting pada balita menggunakan
K-Nearest Neighbor dengan Feature Selection Backward Elimination. *Jurnal Media Informatika
Budidarma*, 6(1), 49. https://doi.org/10.30865/mib.v6i1.3312

Panigoro. (2024). Klasifikasi stunting pada balita dengan algoritma Random Forest dan Support
Vector Machine. *Jurasik (Jurnal Riset Sistem Informasi dan Teknik Informatika)*.
https://tunasbangsa.ac.id/ejurnal/index.php/jurasik/article/view/904

Titimeidara, M. Y., & Hadikurniawati, W. (2021). *Implementasi metode Naive Bayes Classifier
untuk klasifikasi status gizi stunting pada balita* [Laporan penelitian].

---

## 10. Lampiran (Opsional)

- Dataset mentah: `data/data_balita.csv`
- Dataset hasil olahan (setelah cleaning): dihasilkan otomatis oleh notebook (variabel
  `df_clean`), dapat diekspor ke CSV tambahan bila diperlukan.
- Grafik tambahan: seluruh visualisasi (distribusi kelas, histogram, scatter plot, grafik
  performa K, confusion matrix, bar chart perbandingan model) tersedia pada `uas_model.ipynb`.
