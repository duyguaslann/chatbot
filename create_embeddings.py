"""
LangChain + ChromaDB entegrasyonu.
Vectorstore, retriever ve dokuman ekleme islemleri burada.
"""
import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

CHROMADB_HOST   = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT   = int(os.getenv("CHROMADB_PORT", "8000"))
COLLECTION_NAME = "company_data"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

_embeddings  = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        _vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def get_retriever(k: int = 3):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def add_texts(texts: list) -> int:
    get_vectorstore().add_texts(texts)
    return len(texts)


def get_all_documents() -> list:
    results = get_vectorstore()._collection.get()
    return [
        {"id": doc_id, "text": doc}
        for doc_id, doc in zip(results["ids"], results["documents"])
    ]


def delete_documents(ids: list) -> None:
    get_vectorstore().delete(ids=ids)


def update_document(doc_id: str, new_text: str) -> None:
    get_vectorstore().delete(ids=[doc_id])
    get_vectorstore().add_texts([new_text], ids=[doc_id])
