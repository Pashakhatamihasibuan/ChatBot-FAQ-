# src/indexer.py
# Pipeline INDEXING: Baca PDF → Chunking → Embedding → Simpan ke ChromaDB
# Jalankan sekali saat pertama setup, atau setiap ada dokumen baru

import os
import fitz  # PyMuPDF
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    VECTORSTORE_PATH, COLLECTION_NAME
)


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Baca teks dari file PDF menggunakan PyMuPDF.
    Kembalikan dict dengan teks dan metadata.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    total_pages = len(doc)  # Ambil sebelum close
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        full_text += f"\n[Halaman {page_num}]\n{text}"
    doc.close()

    return {
        "text": full_text,
        "source": Path(pdf_path).name,
        "path": pdf_path,
        "total_pages": total_pages
    }


def load_documents_from_folder(folder_path: str) -> list[dict]:
    """
    Baca semua PDF dari folder data/.
    Kembalikan list of document dicts.
    """
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"[!] Tidak ada file PDF ditemukan di: {folder_path}")
        print("    Letakkan file PDF dokumen kampus ke folder 'data/'")
        return []

    documents = []
    for pdf_file in pdf_files:
        print(f"  Membaca: {pdf_file.name}")
        try:
            doc = extract_text_from_pdf(str(pdf_file))
            documents.append(doc)
        except Exception as e:
            print(f"  [ERROR] Gagal membaca {pdf_file.name}: {e}")

    print(f"\n[OK] Berhasil membaca {len(documents)} dokumen PDF")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Potong dokumen menjadi chunk-chunk kecil.
    Ini adalah tahap krusial dalam RAG — ukuran chunk sangat mempengaruhi kualitas.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritas pemisahan
        length_function=len,
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": doc["source"],
                    "chunk_id": i,
                    "chunk_size_config": CHUNK_SIZE,
                }
            })

    print(f"[OK] Total chunk dihasilkan: {len(all_chunks)}")
    print(f"     Rata-rata panjang chunk: {sum(len(c['text']) for c in all_chunks) // len(all_chunks)} karakter")
    return all_chunks


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """
    Buat embedding dari setiap chunk dan simpan ke ChromaDB.
    Proses ini mungkin butuh beberapa menit pertama kali (download model).
    """
    print(f"\n[...] Memuat embedding model: {EMBEDDING_MODEL}")
    print("      (Download ~120MB saat pertama kali, tunggu sebentar)")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Penting untuk cosine similarity
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"[...] Membuat embedding untuk {len(texts)} chunks...")

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_PATH,
    )

    print(f"[OK] Vector database tersimpan di: {VECTORSTORE_PATH}/")
    return vectorstore


def load_vectorstore() -> Chroma:
    """
    Muat vector database yang sudah ada (tidak perlu indexing ulang).
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTORSTORE_PATH,
    )
    return vectorstore


def run_indexing(data_folder: str = "./data"):
    """
    Jalankan pipeline indexing lengkap dari awal.
    """
    print("=" * 50)
    print("   PIPELINE INDEXING RAG KAMPUS")
    print("=" * 50)

    # Step 1: Baca PDF
    print("\n[1/3] Membaca dokumen PDF...")
    documents = load_documents_from_folder(data_folder)
    if not documents:
        return None

    # Step 2: Chunking
    print(f"\n[2/3] Memotong dokumen (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents)

    # Step 3: Embedding + simpan
    print("\n[3/3] Membuat embedding dan menyimpan ke ChromaDB...")
    vectorstore = build_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("   INDEXING SELESAI!")
    print(f"   {len(documents)} dokumen | {len(chunks)} chunks tersimpan")
    print("=" * 50)
    return vectorstore


if __name__ == "__main__":
    run_indexing(data_folder="./data")
