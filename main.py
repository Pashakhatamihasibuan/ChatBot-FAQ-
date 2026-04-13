# main.py
# Entry point untuk menjalankan sistem RAG via command line
# Gunakan ini untuk testing sebelum pakai UI Streamlit

import argparse
from pathlib import Path
from src.indexer import run_indexing, load_vectorstore
from src.rag_chain import build_rag_chain, ask_question
from src.config import VECTORSTORE_PATH, GEMINI_API_KEY


def check_api_key():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyBgKwFs7lzPmJxKtPbX1wKuyiGRfDzaAbk":
        print("[ERROR] GEMINI_API_KEY belum diset!")
        print("  1. Salin .env.example menjadi .env")
        print("  2. Isi API key dari https://aistudio.google.com/app/apikey")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot FAQ Kampus")
    parser.add_argument("--index", action="store_true",
                        help="Jalankan pipeline indexing dokumen PDF")
    parser.add_argument("--chat", action="store_true",
                        help="Mode chat interaktif di terminal")
    parser.add_argument("--evaluate", action="store_true",
                        help="Jalankan evaluasi RAGAS")
    parser.add_argument("--ask", type=str,
                        help="Ajukan satu pertanyaan langsung")

    args = parser.parse_args()

    # Default: tampilkan help jika tidak ada argumen
    if not any(vars(args).values()):
        parser.print_help()
        print("\nContoh penggunaan:")
        print("  python main.py --index          # Indexing dokumen")
        print("  python main.py --chat           # Mode chat")
        print("  python main.py --ask 'Syarat wisuda?'")
        print("  python main.py --evaluate       # Evaluasi RAGAS")
        print("  streamlit run app.py            # Buka UI chatbot")
        return

    # ─── Mode: Indexing ───────────────────────────────────
    if args.index:
        print("Memulai pipeline indexing...")
        run_indexing(data_folder="./data")
        return

    # ─── Load vectorstore (diperlukan untuk chat/evaluate/ask) ───
    check_api_key()

    vectorstore_exists = Path(VECTORSTORE_PATH).exists() and \
                         any(Path(VECTORSTORE_PATH).iterdir())
    if not vectorstore_exists:
        print("[!] Vector database belum ada. Jalankan indexing dulu:")
        print("    python main.py --index")
        return

    print("[...] Memuat sistem RAG...")
    vectorstore = load_vectorstore()
    rag_chain, retriever = build_rag_chain(vectorstore)
    print("[OK] Sistem RAG siap!\n")

    # ─── Mode: Satu pertanyaan ─────────────────────────────
    if args.ask:
        result = ask_question(rag_chain, retriever, args.ask)
        print(f"Pertanyaan: {result['question']}")
        print(f"\nJawaban:\n{result['answer']}")
        print(f"\nSumber ({result['num_chunks_retrieved']} chunk):")
        for src in result["sources"]:
            print(f"  - {src['source']} (chunk #{src['chunk_id']})")
        return

    # ─── Mode: Chat interaktif ─────────────────────────────
    if args.chat:
        print("=" * 50)
        print("   CHATBOT FAQ AKADEMIK KAMPUS")
        print("   Ketik 'keluar' untuk berhenti")
        print("=" * 50)

        while True:
            question = input("\nPertanyaan: ").strip()
            if question.lower() in ["keluar", "exit", "quit"]:
                print("Terima kasih, sampai jumpa!")
                break
            if not question:
                continue

            result = ask_question(rag_chain, retriever, question)
            print(f"\nJawaban:\n{result['answer']}")
            print(f"\n[Sumber: {', '.join(set(s['source'] for s in result['sources']))}]")

    # ─── Mode: Evaluasi ────────────────────────────────────
    if args.evaluate:
        from src.evaluator import run_evaluation
        df = run_evaluation(rag_chain, retriever)
        print("\nEvaluasi selesai! Cek file hasil_evaluasi.csv")


if __name__ == "__main__":
    main()
