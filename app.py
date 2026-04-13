# app.py
# UI Chatbot menggunakan Streamlit
# Jalankan dengan: streamlit run app.py

import streamlit as st
import os
from pathlib import Path
from src.indexer import run_indexing, load_vectorstore
from src.rag_chain import build_rag_chain, ask_question
from src.config import VECTORSTORE_PATH, GEMINI_API_KEY

# ─── Konfigurasi Halaman ───────────────────────────────────
st.set_page_config(
    page_title="FAQ Akademik Kampus",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Chatbot FAQ Akademik Kampus")
st.caption("Tanya apa saja seputar peraturan akademik, wisuda, kurikulum, dan prosedur kampus.")


# ─── Inisialisasi Sistem ───────────────────────────────────
@st.cache_resource(show_spinner="Memuat sistem RAG...")
def initialize_rag():
    """Cache sistem RAG supaya tidak di-load ulang setiap interaksi."""
    if not GEMINI_API_KEY:
        st.error("❌ GEMINI_API_KEY belum diset! Salin .env.example ke .env dan isi API key.")
        st.stop()

    vectorstore_exists = Path(VECTORSTORE_PATH).exists() and \
                         any(Path(VECTORSTORE_PATH).iterdir())

    if vectorstore_exists:
        vectorstore = load_vectorstore()
    else:
        st.info("📄 Vector database belum ada. Menjalankan indexing dokumen...")
        vectorstore = run_indexing(data_folder="./data")
        if vectorstore is None:
            st.error("❌ Indexing gagal. Pastikan ada file PDF di folder data/")
            st.stop()

    rag_chain, retriever = build_rag_chain(vectorstore)
    return rag_chain, retriever


# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    st.markdown("**Dokumen yang tersedia:**")
    data_folder = Path("./data")
    pdf_files = list(data_folder.glob("*.pdf")) if data_folder.exists() else []
    if pdf_files:
        for f in pdf_files:
            st.markdown(f"- 📄 {f.name}")
    else:
        st.warning("Belum ada dokumen PDF di folder data/")

    st.divider()

    if st.button("🔄 Rebuild Index", help="Jalankan ulang indexing jika ada dokumen baru"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("**Contoh pertanyaan:**")
    example_questions = [
        "Apa syarat wisuda S1?",
        "Bagaimana cara mengajukan cuti akademik?",
        "Berapa SKS minimal untuk lulus?",
        "Apa itu KRS dan kapan batas pengisiannya?",
    ]
    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["pending_question"] = q


# ─── Chat Interface ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sumber dokumen"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** (chunk #{src['chunk_id']})")
                    st.caption(src["preview"])

# Handle pertanyaan dari sidebar button
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")
else:
    user_input = st.chat_input("Ketik pertanyaanmu di sini...")

if user_input:
    # Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate jawaban
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi..."):
            try:
                rag_chain, retriever = initialize_rag()
                result = ask_question(rag_chain, retriever, user_input)

                st.markdown(result["answer"])

                with st.expander(f"📎 Sumber dokumen ({result['num_chunks_retrieved']} chunk)"):
                    for src in result["sources"]:
                        st.markdown(f"**{src['source']}** (chunk #{src['chunk_id']})")
                        st.caption(src["preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as e:
                error_msg = f"❌ Terjadi kesalahan: {str(e)}"
                st.error(error_msg)
