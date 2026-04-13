# src/config.py
# Semua konfigurasi sistem RAG ada di sini
# Ubah nilai-nilai ini untuk eksperimen di tesis kamu

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Key ───────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── Embedding Model ───────────────────────────────────────
# Model multilingual yang support Bahasa Indonesia
# Alternatif: "firqaaa/indo-sentence-bert-base" (khusus Indo)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# ─── Chunking Strategy ─────────────────────────────────────
# Ini salah satu variabel eksperimen tesis kamu!
# Coba bandingkan: CHUNK_SIZE = 256 vs 512 vs 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # Overlap antar chunk supaya konteks tidak putus

# ─── Retrieval Settings ────────────────────────────────────
# Berapa chunk yang diambil per pertanyaan
TOP_K_RESULTS = 4

# ─── Vector Database ───────────────────────────────────────
VECTORSTORE_PATH = "./vectorstore"
COLLECTION_NAME = "dokumen_kampus"

# ─── LLM Settings ──────────────────────────────────────────
# Ollama: jalankan model lokal, tidak perlu internet & API key
LLM_PROVIDER = "ollama"        # Ganti ke "gemini" kalau mau balik ke Gemini
LLM_MODEL = "gemma4:26b"         # Nama model sesuai hasil: ollama list
LLM_BASE_URL = "http://localhost:11434"  # URL Ollama default
LLM_TEMPERATURE = 0.1


# ─── Prompt Template ───────────────────────────────────────
# Prompt ini penting! Instruksikan LLM untuk HANYA menjawab dari konteks
SYSTEM_PROMPT = """Kamu adalah asisten informasi akademik kampus yang membantu mahasiswa.
Jawab pertanyaan mahasiswa HANYA berdasarkan konteks dokumen yang diberikan.
Jika informasi tidak ada dalam konteks, katakan dengan jelas bahwa kamu tidak menemukan informasinya dalam dokumen.
Gunakan bahasa Indonesia yang sopan dan mudah dipahami.
Sertakan referensi sumber dokumen di akhir jawaban jika relevan.

Konteks dokumen:
{context}

Pertanyaan mahasiswa: {question}

Jawaban:"""
