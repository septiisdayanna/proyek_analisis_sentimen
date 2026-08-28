# SeaBank App Review Sentiment Analysis

## Overview
Project ini membangun model analisis sentimen untuk mengklasifikasikan ulasan pengguna aplikasi **SeaBank** (bank digital) di Google Play Store ke dalam tiga kategori: **Negatif**, **Netral**, dan **Positif**. Seluruh data merupakan hasil scraping mandiri, diolah melalui pipeline NLP Bahasa Indonesia, lalu dilatih menggunakan model klasifikasi.

## Background / Problem
Ulasan pengguna di Play Store adalah sumber masukan yang sangat berharga bagi pengembang aplikasi, namun jumlahnya bisa sangat banyak dan sulit dibaca satu per satu. Project ini mencoba mengotomasi pemahaman sentimen pengguna terhadap aplikasi SeaBank, sehingga tren keluhan atau kepuasan pengguna dapat terlihat secara terukur, bukan hanya berdasarkan kesan sekilas.

## Objectives
- Mengumpulkan data ulasan aplikasi SeaBank secara mandiri melalui web scraping.
- Melakukan ekstraksi fitur dan pelabelan sentimen pada data ulasan.
- Melatih model klasifikasi sentimen dengan target akurasi testing minimal 85%.
- Menguji beberapa skema pelatihan (algoritma & pendekatan berbeda) untuk membandingkan performa.

## Features
- **Scraping mandiri** — mengambil hingga 35.000 ulasan aplikasi SeaBank langsung dari Google Play Store menggunakan `google_play_scraper`.
- **Preprocessing teks Bahasa Indonesia** — meliputi pembersihan teks (URL, mention, angka, tanda baca), case folding, normalisasi kata gaul/singkatan menggunakan kamus custom (`combined_slang_words.txt` berisi ratusan pasangan kata slang → baku), tokenisasi, dan penghapusan stopword (Bahasa Indonesia + Inggris + daftar kustom tambahan).
- **Ekstraksi fitur** — representasi teks menggunakan TF-IDF.
- **Klasifikasi 3 kelas** — sentimen dikategorikan menjadi Negatif, Netral, dan Positif.
- **Perbandingan skema pelatihan** — mencoba lebih dari satu pendekatan (machine learning klasik dan deep learning) untuk menemukan kombinasi terbaik.
- **Inference siap pakai** — fungsi `predict_sentiment()` menerima teks baru, menjalankannya melalui seluruh pipeline preprocessing yang sama seperti saat training, lalu mengembalikan label sentimen.

## Tech Stack
- **Bahasa:** Python
- **Scraping:** google-play-scraper
- **NLP preprocessing:** NLTK (tokenizing, stopwords), emoji, regex
- **Machine Learning:** scikit-learn (TF-IDF Vectorizer, Logistic Regression, Decision Tree, Naive Bayes, Random Forest, Support Vector Machine(SVM))
- **Deep Learning:** *(LSTM, GRU)*
- **Model persistence:** joblib

## How It Works / Methodology

**1. Data Collection** (`scraping_dataset.ipynb`)
Mengambil hingga 35.000 ulasan aplikasi SeaBank (`id.co.bankbkemobile.digitalbank`) dari Google Play Store Indonesia, diurutkan berdasarkan relevansi, lalu disimpan sebagai `ulasan_aplikasi.csv`.

**2. Preprocessing & Pelabelan** (`notebook.ipynb`)
Berdasarkan struktur data dan alur pemrosesan yang ada pada `notebook.ipynb`, berikut adalah rincian untuk melengkapi poin **2 (Preprocessing & Pelabelan)** dan **3 (Ekstraksi Fitur & Training)**:

#### **Metode Pelabelan Sentimen**

Pelabelan sentimen pada dataset dilakukan secara **otomatis berbasis skor rating (Rating-Based Labeling)** dari ulasan Google Play Store (`score` 1 hingga 5):

* **Positif**: Ulasan dengan rating **4** atau **5**.
* **Negatif**: Ulasan dengan rating **1** atau **2**.
* **Netral**: Ulasan dengan rating **3** *(atau dialokasikan terpisah sesuai pemetaan kelas yang digunakan)*.

#### **Tahapan Preprocessing Teks**

Proses pembersihan dan normalisasi teks dilakukan secara sistematis dalam beberapa tahapan:

1. **Cleaning Text**: Menghapus emoji (`emoji.replace_emoji`), *mention* (`@`), *hashtag* (`#`), URL (`http/https`), angka, karakter khusus/tanda baca (`string.punctuation`), serta spasi berlebih.
2. **Case Folding**: Mengubah seluruh huruf dalam teks menjadi huruf kecil (*lowercase*).
3. **Slang Word Normalization**: Mengubah kata-kata tidak baku/gaul menjadi kata baku dalam bahasa Indonesia menggunakan kamus kata gaul (misal: `"tdk"` $\rightarrow$ `"tidak"`, `"lemot"` $\rightarrow$ `"lambat"`, `"dgn"` $\rightarrow$ `"dengan"`).
4. **Stopword Removal & Tokenization**: Menghapus kata-kata umum yang tidak memberikan nilai informasi tinggi menggunakan daftar *stopwords* dari NLTK / Sastrawi, lalu membagi teks menjadi token/kata individual.
5. **Stemming**: Mengubah kata berimbuhan menjadi kata dasar menggunakan `Sastrawi.Stemmer.StemmerFactory` (diakselerasi dengan `swifter`).

---

### **Skema Pelatihan & Algoritma yang Dicoba**

Proses pemodelan menggunakan representasi numerik **TF-IDF (Term Frequency - Inverse Document Frequency)** untuk mengekstraksi fitur teks, kemudian dilatih dan dievaluasi menggunakan tiga skema pemodelan/eksperimen utama:

1. **Skema 1: Machine Learning Klasik Baseline (Linear & Probabilistic Models)**
* **Algoritma**: *Logistic Regression* dan *Bernoulli Naive Bayes* (`BernoulliNB`).
* **Tujuan**: Menguji performa model linier dan probabilistik dasar dalam melakukan klasifikasi teks dengan representasi sparse TF-IDF.


2. **Skema 2: Tree-Based & Support Vector Models**
* **Algoritma**: *Decision Tree Classifier*, *Random Forest Classifier*, dan *Support Vector Classifier* (`SVC`).
* **Tujuan**: Membandingkan performa algoritma berbasis *decision tree* / *ensemble learning* serta pemisah hyperplane nonlinear (*SVM*) terhadap variasi bobot kata.


3. **Skema 3: Evaluasi & Perbandingan Multi-Model**
* **Proses**: Membagi dataset menjadi *training set* dan *testing set* (menggunakan `train_test_split`), mengukur dan membandingkan metrik *Accuracy*, *Precision*, *Recall*, dan *F1-Score* di seluruh model yang diuji untuk menemukan model terbaik (*best performing model*) sebelum disimpan (`joblib`).

**3. Ekstraksi Fitur & Training**
Teks yang sudah bersih diubah menjadi representasi numerik menggunakan TF-IDF, lalu digunakan untuk melatih model klasifikasi. Beberapa skema pelatihan diuji untuk membandingkan performa, termasuk model machine learning klasik (Logistic Regression) dan deep learning.

**4. Inference** (`inference.ipynb`)
Fungsi `predict_sentiment()` menjalankan teks input baru melalui pipeline preprocessing yang identik dengan saat training (cleaning → case folding → normalisasi slang → tokenisasi → filtering stopword), mengubahnya ke representasi TF-IDF, lalu memprediksi label sentimen menggunakan model Logistic Regression yang tersimpan (`logistic_regression.pkl` + `tfidf_vectorizer.pkl`).

## Dataset
- **Sumber:** Ulasan aplikasi SeaBank Indonesia di Google Play Store (scraping mandiri)
- **Target jumlah data:** 35.000 ulasan (`ulasan_aplikasi.csv`), berkurang setelah tahap pembersihan menjadi `cleaned_data_fix.csv`
- **Kelas sentimen:** Negatif, Netral, Positif

## Results / Output
Berdasarkan hasil review:
- Model berhasil mencapai target minimum akurasi testing **≥85%** pada kriteria wajib.
- Namun, dari seluruh skema pelatihan yang diuji, **belum ada yang mencapai akurasi 92%** pada training maupun testing set.
- Hanya **2 dari 3 skema pelatihan** yang berhasil mencapai akurasi di atas 85% pada train **dan** test set sekaligus.

## Installation
```bash
git clone https://github.com/septiisdayanna/proyek_analisis_sentimen.git
cd proyek_analisis_sentimen
pip install -r requirements.txt
```

## Usage
Melakukan scraping ulang data ulasan (opsional, data hasil scraping sudah tersedia di `ulasan_aplikasi.csv`):
```bash
jupyter notebook scraping_dataset.ipynb
```

Menjalankan prediksi sentimen pada teks baru:
```python
from inference import predict_sentiment
import joblib

model = joblib.load('logistic_regression.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

predict_sentiment("Aplikasinya bagus banget, transfer cepat dan gratis!", model, tfidf)
# Output: "Positif"
```

## Project Structure
```
proyek_analisis_sentimen/
├── scraping_dataset.ipynb        # Scraping ulasan dari Google Play Store
├── notebook.ipynb                # EDA, pelabelan, ekstraksi fitur, training (beberapa skema)
├── inference.ipynb               # Fungsi prediksi sentimen siap pakai
├── combined_slang_words.txt      # Kamus normalisasi kata gaul/singkatan Bahasa Indonesia
├── ulasan_aplikasi.csv           # Data mentah hasil scraping
├── cleaned_data_fix.csv          # Data setelah preprocessing & pelabelan
├── logistic_regression.pkl       # Model terlatih (skema terbaik)
├── tfidf_vectorizer.pkl          # Vectorizer TF-IDF terlatih
└── requirements.txt
```

## Limitations
- **Distribusi kelas sentimen tidak seimbang** — sesuai temuan reviewer, dataset hasil pelabelan memiliki jumlah data yang tidak merata antar kelas, yang berisiko membuat model bias ke kelas mayoritas.
- Akurasi terbaik yang dicapai masih di bawah 92% pada seluruh skema pelatihan yang diuji.
- Metode pelabelan sentimen (lexicon-based atau lainnya) berpotensi mengandung bias, terutama pada teks dengan sarkasme atau konteks ambigu yang umum pada ulasan aplikasi.
- Preprocessing masih menggunakan pendekatan ekstraksi fitur tradisional (TF-IDF), belum mengeksplorasi representasi berbasis pretrained model (Word2Vec, FastText, atau IndoBERT) yang berpotensi menangkap makna kontekstual lebih baik untuk Bahasa Indonesia.

## Future Improvements
- Menangani class imbalance melalui oversampling (SMOTE), undersampling, atau `class_weight` saat training, sesuai saran reviewer.
- Mencoba ekstraksi fitur berbasis pretrained model (misalnya IndoBERT) untuk menangkap konteks bahasa yang lebih kaya dibanding TF-IDF.
- Melakukan hyperparameter tuning secara sistematis pada model yang dipilih.
- Meninjau ulang proses pelabelan untuk memastikan konsistensi dan mengurangi potensi bias, terutama pada kalimat ambigu.

## Author
**Septi Isdayanna**
