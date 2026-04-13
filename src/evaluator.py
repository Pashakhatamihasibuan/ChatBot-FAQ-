# src/evaluator.py
# Evaluasi sistem RAG menggunakan framework RAGAS
# Output-nya adalah metrik untuk BAB 4 tesis kamu!

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Apakah jawaban sesuai dengan dokumen sumber?
    answer_relevancy,      # Apakah jawaban relevan dengan pertanyaan?
    context_recall,        # Apakah chunk yang diambil mengandung info yang dibutuhkan?
    context_precision,     # Apakah chunk yang diambil memang relevan semua?
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import GEMINI_API_KEY, LLM_MODEL, EMBEDDING_MODEL


# ─── Dataset Evaluasi ──────────────────────────────────────
# Buat 50-100 pasang QA ini secara manual dari dokumen kampusmu
# ground_truth = jawaban yang BENAR (kamu tulis sendiri dari dokumen)
EVAL_DATASET = [
    {
        "question": "Apa saja syarat untuk mengajukan sidang skripsi?",
        "ground_truth": "Mahasiswa harus menyelesaikan minimal 144 SKS, IPK minimal 2.00, dan telah menyelesaikan seminar proposal."
    },
    {
        "question": "Bagaimana prosedur pengajuan cuti akademik?",
        "ground_truth": "Mahasiswa mengajukan permohonan tertulis kepada dekan dengan melampirkan alasan dan persetujuan orang tua/wali, paling lambat 2 minggu sebelum semester dimulai."
    },
    {
        "question": "Berapa batas maksimal masa studi S1?",
        "ground_truth": "Batas maksimal masa studi program S1 adalah 7 tahun atau 14 semester."
    },
    # Tambahkan 47+ pasang QA lainnya dari dokumen kampusmu
    # Semakin banyak = evaluasi semakin valid untuk tesis
]


def prepare_eval_data(rag_chain, retriever, eval_dataset: list) -> Dataset:
    """
    Jalankan semua pertanyaan evaluasi melalui sistem RAG,
    kumpulkan jawaban dan konteks yang digunakan.
    """
    print(f"[...] Menjalankan {len(eval_dataset)} pertanyaan evaluasi...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(eval_dataset, start=1):
        q = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i}/{len(eval_dataset)}] {q[:60]}...")

        # Ambil konteks
        retrieved_docs = retriever.invoke(q)
        ctx = [doc.page_content for doc in retrieved_docs]

        # Generate jawaban
        answer = rag_chain.invoke(q)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_evaluation(rag_chain, retriever, eval_dataset=None) -> pd.DataFrame:
    """
    Jalankan evaluasi RAGAS lengkap.
    Kembalikan DataFrame dengan semua metrik — ini yang masuk BAB 4!
    """
    if eval_dataset is None:
        eval_dataset = EVAL_DATASET

    print("=" * 50)
    print("   EVALUASI SISTEM RAG")
    print("=" * 50)

    # Siapkan data evaluasi
    dataset = prepare_eval_data(rag_chain, retriever, eval_dataset)

    # LLM dan embedding untuk RAGAS (evaluator)
    eval_llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )
    eval_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\n[...] Menghitung metrik RAGAS (butuh beberapa menit)...")

    # Hitung semua metrik
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,      # Seberapa faktual jawaban terhadap dokumen
            answer_relevancy,  # Seberapa relevan jawaban terhadap pertanyaan
            context_recall,    # Seberapa lengkap konteks yang diambil
            context_precision, # Seberapa presisi konteks yang diambil
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    # Tampilkan hasil
    df = results.to_pandas()

    print("\n" + "=" * 50)
    print("   HASIL EVALUASI (RATA-RATA)")
    print("=" * 50)
    print(f"  Faithfulness       : {df['faithfulness'].mean():.4f}  (0-1, semakin tinggi semakin baik)")
    print(f"  Answer Relevancy   : {df['answer_relevancy'].mean():.4f}  (0-1)")
    print(f"  Context Recall     : {df['context_recall'].mean():.4f}  (0-1)")
    print(f"  Context Precision  : {df['context_precision'].mean():.4f}  (0-1)")
    print("=" * 50)

    # Simpan hasil ke CSV untuk tesis
    output_path = "hasil_evaluasi.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Hasil lengkap tersimpan di: {output_path}")

    return df


if __name__ == "__main__":
    # Untuk menjalankan evaluasi, import chain dari main.py
    print("Jalankan evaluasi melalui: python main.py --evaluate")
