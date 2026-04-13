# RAG Chatbot FAQ Akademik Kampus
> Implementasi Retrieval-Augmented Generation untuk Sistem Tanya-Jawab Dokumen Akademik  
> Tesis — Pendidikan Teknik Elektronika dan Informatika

---

## Struktur Project

```
rag_kampus/
├── data/                  ← Letakkan file PDF dokumen kampus di sini
├── vectorstore/           ← Dibuat otomatis saat indexing
├── src/
│   ├── config.py          ← Semua konfigurasi (chunk size, model, dsb)
│   ├── indexer.py         ← Pipeline indexing: PDF → ChromaDB
│   ├── rag_chain.py       ← Pipeline querying: Pertanyaan → Jawaban
│   └── evaluator.py       ← Evaluasi RAGAS untuk metrik tesis
├── app.py                 ← UI chatbot Streamlit
├── main.py                ← CLI entry point
├── requirements.txt
└── .env.example
```

---

## Setup (Langkah Demi Langkah)

### 1. Buat virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows
```

### 2. Install dependensi
```bash
pip install -r requirements.txt
```

### 3. Setup API Key
```bash
cp .env.example .env
```
Buka file `.env`, ganti dengan API key Gemini kamu:  
Dapatkan gratis di → https://aistudio.google.com/app/apikey

### 4. Siapkan dokumen PDF
Letakkan file PDF dokumen kampus ke folder `data/`:
- `panduan_akademik.pdf`
- `peraturan_mahasiswa.pdf`
- `prosedur_wisuda.pdf`
- dst.

### 5. Jalankan indexing
```bash
python main.py --index
```
Proses ini akan:
- Membaca semua PDF dari folder `data/`
- Memotong teks menjadi chunks
- Membuat embedding (download model ~120MB pertama kali)
- Menyimpan ke ChromaDB di folder `vectorstore/`

---

## Cara Menggunakan

### UI Chatbot (Recommended)
```bash
streamlit run app.py
```
Buka browser → http://localhost:8501

### CLI — satu pertanyaan
```bash
python main.py --ask "Apa syarat untuk mengajukan sidang skripsi?"
```

### CLI — mode chat interaktif
```bash
python main.py --chat
```

### Evaluasi RAGAS
```bash
python main.py --evaluate
```
Hasil evaluasi tersimpan di `hasil_evaluasi.csv`

---

## Variabel Eksperimen untuk Tesis

Di `src/config.py`, ubah nilai ini dan bandingkan hasilnya:

| Parameter | Nilai yang Bisa Dicoba | Pengaruh |
|---|---|---|
| `CHUNK_SIZE` | 256, 512, 1024 | Granularitas informasi |
| `CHUNK_OVERLAP` | 0, 64, 128 | Kontinuitas antar chunk |
| `TOP_K_RESULTS` | 2, 4, 6 | Jumlah konteks ke LLM |
| `EMBEDDING_MODEL` | multilingual-e5-small vs indo-sentence-bert | Kualitas embedding Bahasa Indonesia |

Setiap kombinasi → jalankan `--evaluate` → catat metrik RAGAS → bandingkan di BAB 4.

---

## Metrik Evaluasi (RAGAS)

| Metrik | Penjelasan |
|---|---|
| **Faithfulness** | Apakah jawaban sesuai dengan dokumen sumber? |
| **Answer Relevancy** | Apakah jawaban menjawab pertanyaan? |
| **Context Recall** | Apakah semua informasi penting berhasil diambil? |
| **Context Precision** | Apakah chunk yang diambil memang relevan? |

---

## Stack Teknologi

| Komponen | Library | Versi |
|---|---|---|
| Orkestrasi RAG | LangChain | 0.2.x |
| LLM | Gemini 1.5 Flash (Google) | via API |
| Embedding | multilingual-e5-small (HuggingFace) | — |
| Vector DB | ChromaDB | 0.5.x |
| UI | Streamlit | 1.38.x |
| Evaluasi | RAGAS | 0.1.x |
| PDF Parser | PyMuPDF | 1.24.x |
