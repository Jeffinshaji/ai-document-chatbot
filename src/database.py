"""
database.py — FAISS HNSW vector store: create, load, update, and query sources.
"""

import os
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)

# ── Config values ────────────────────────────────────────────────
_path_cfg   = config["paths"]
_vs_cfg     = config["vectorstore"]
_emb_cfg    = config["embeddings"]

INDEX_PATH  = _path_cfg["faiss_index"]
HNSW_M      = _vs_cfg["hnsw_m"]
SAMPLE_TEXT = _emb_cfg["sample_text"]
EMB_MODEL   = _emb_cfg["model_name"]


def create_vector_store(documents):
    """
    Creates or loads a FAISS HNSW vector store.
    - First run  : builds index from scratch and saves to disk.
    - Later runs : loads from disk and adds new documents.

    Args:
        documents: List of LangChain Document objects (chunked).

    Returns:
        FAISS vectorstore instance.
    """
    logger.info("Initialising vector store | index_path=%s", INDEX_PATH)
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    logger.debug("Embeddings model loaded: %s", EMB_MODEL)

    # ── Load existing index from disk ────────────────────────────
    if os.path.exists(INDEX_PATH):
        logger.info("Existing FAISS index found at '%s'. Loading from disk...", INDEX_PATH)
        try:
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully. Adding %d new document(s)...", len(documents))
            vectorstore.add_documents(documents)
            vectorstore.save_local(INDEX_PATH)
            logger.info("Index updated and saved to '%s'.", INDEX_PATH)
            return vectorstore
        except Exception as e:
            logger.error("Failed to load existing FAISS index: %s. Rebuilding from scratch.", e)

    # ── First time — build from scratch ──────────────────────────
    logger.info("No existing index found. Building new FAISS HNSW index from scratch...")

    try:
        dimension = len(embeddings.embed_query(SAMPLE_TEXT))
        logger.debug("Embedding dimension detected: %d", dimension)
    except Exception as e:
        logger.error("Failed to detect embedding dimension: %s", e)
        raise

    # HNSW index — fast and accurate for small-medium datasets
    index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_L2)
    logger.debug("FAISS IndexHNSWFlat created | dimension=%d | M=%d", dimension, HNSW_M)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    logger.info("Generating embeddings and indexing %d document(s) with HNSW...", len(documents))
    try:
        vectorstore.add_documents(documents)
    except Exception as e:
        logger.error("Failed to add documents to FAISS index: %s", e)
        raise

    vectorstore.save_local(INDEX_PATH)
    logger.info("New FAISS index saved to '%s'.", INDEX_PATH)

    return vectorstore


def get_available_sources(vectorstore) -> list[str]:
    """
    Returns a sorted list of unique source filenames stored in the index.
    Useful for metadata filtering — shows what files are available.

    Args:
        vectorstore: FAISS vectorstore instance.

    Returns:
        Sorted list of unique source filename strings.
    """
    logger.debug("Fetching available sources from vectorstore docstore.")
    sources = set()
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if doc and hasattr(doc, "metadata"):
            source = doc.metadata.get("source", "")
            if source:
                sources.add(os.path.basename(source))

    result = sorted(list(sources))
    logger.debug("Available sources (%d): %s", len(result), result)
    return result
