# src/rag_chain.py
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL,
    TOP_K_RESULTS, SYSTEM_PROMPT, EMBEDDING_MODEL,
    VECTORSTORE_PATH, COLLECTION_NAME,
    GEMINI_API_KEY
)


def get_llm():
    """
    Buat LLM berdasarkan LLM_PROVIDER di config.py.
    Ganti LLM_PROVIDER = "ollama" atau "gemini" sesuai kebutuhan.
    """
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
    else:
        raise ValueError(f"LLM_PROVIDER tidak dikenal: {LLM_PROVIDER}. Pilih 'ollama' atau 'gemini'.")


def format_retrieved_docs(docs) -> str:
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Dokumen tidak diketahui")
        formatted.append(f"[Sumber {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def load_vectorstore() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTORSTORE_PATH,
    )


def build_rag_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT
    )

    llm = get_llm()

    rag_chain = (
        {
            "context": retriever | format_retrieved_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask_question(rag_chain, retriever, question: str, max_retries: int = 3) -> dict:
    retrieved_docs = retriever.invoke(question)

    for attempt in range(max_retries):
        try:
            answer = rag_chain.invoke(question)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 35 * (attempt + 1)
                print(f"Rate limit, tunggu {wait} detik...")
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    sources = []
    for doc in retrieved_docs:
        source_info = {
            "source": doc.metadata.get("source", "Unknown"),
            "chunk_id": doc.metadata.get("chunk_id", "-"),
            "preview": doc.page_content[:150] + "..."
        }
        if source_info not in sources:
            sources.append(source_info)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_chunks_retrieved": len(retrieved_docs)
    }
