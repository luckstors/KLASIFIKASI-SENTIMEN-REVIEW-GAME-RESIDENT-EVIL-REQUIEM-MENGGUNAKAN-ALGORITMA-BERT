# KLASIFIKASI-SENTIMEN-REVIEW-GAME-RESIDENT-EVIL-REQUIEM-MENGGUNAKAN-ALGORITMA-BERT

# Klasifikasi Sentimen Review Steam — Resident Evil 9 Requiem (BERT)

Proyek klasifikasi sentimen (positif/negatif) terhadap review game **Resident Evil Requiem** di Steam, menggunakan model **BERT (bert-base-uncased)** dari HuggingFace Transformers.

| Info | Detail |
|---|---|
| Game | Resident Evil Requiem (Steam App ID: 3764200) |
| Model | `bert-base-uncased` |
| Tujuan | Klasifikasi sentimen review: Positif / Negatif |
| Strategi Data | Balanced 50% Positif : 50% Negatif |

## Dataset

- Sumber: Steam API (`/appreviews/{app_id}`) untuk game **Resident Evil Requiem** (App ID: 3764200)
- Total: 2000 review (1000 positif, 1000 negatif) → 1997 setelah dibersihkan (3 duplikat dihapus)
- Kolom: `review`, `label` (1=Positif, 0=Negatif), `game`

## Alur Proses

1. **Scraping data** — mengambil review dari Steam API secara balanced (positif & negatif)
2. **Preprocessing teks**
   - Hapus URL
   - Hapus tag HTML
   - Hapus karakter non-ASCII (emoji, simbol)
   - Hapus karakter khusus (pertahankan huruf, angka, tanda baca dasar)
   - Normalisasi huruf berulang (contoh: `soooo` → `soo`)
   - Hapus spasi berlebih & lowercase
3. **Tokenisasi** menggunakan `BertTokenizer`
4. **Split data**: 70% Train (1442) / 15% Validasi (255) / 15% Test (300), stratified
5. **Fine-tuning** `BertForSequenceClassification` dengan HuggingFace `Trainer`
   - Epochs: 3, Batch size: 16, Learning rate: 2e-5
6. **Evaluasi**: accuracy, precision, recall, F1-score, confusion matrix
7. **Visualisasi**: distribusi sentimen, WordCloud (keseluruhan, positif, negatif), panjang review per sentimen, kata paling sering muncul
8. **Uji coba manual** dengan kalimat/review custom di luar dataset

## Hasil Evaluasi

| Metrik | Skor |
|---|---|
| Accuracy | 92.67% |
| Precision | 91.03% |
| Recall | 94.67% |
| F1-Score | 92.81% |

**Per kelas:**

| Kelas | Precision | Recall | F1-Score |
|---|---|---|---|
| Negatif | 0.94 | 0.91 | 0.93 |
| Positif | 0.91 | 0.95 | 0.93 |

## Struktur Repo

```
.
├── klasifikasi_sentimen_re9_bert.ipynb   # Notebook utama
├── requirements.txt                       # Dependencies
└── README.md
```

> Catatan: dataset (`data_mentah_steam.csv`) di-generate lewat proses scraping di dalam notebook (sel "Konfigurasi Game & Target Scraping"). Jalankan sel scraping terlebih dahulu, atau siapkan CSV dengan kolom `review`, `label`, `game` sebelum menjalankan sel preprocessing.

## Instalasi

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

1. Buka `klasifikasi_sentimen_re9_bert.ipynb` di Jupyter Notebook / Google Colab
2. Jalankan sel secara berurutan dari atas ke bawah
3. Model akan mengunduh `bert-base-uncased` (~440MB) otomatis saat pertama kali dijalankan

## Author

Muhammad Rafi Fadillah
