"""
processor.py — Splits loaded documents into overlapping chunks
               for embedding and indexing.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)

# ── Config values ────────────────────────────────────────────────
_splitter_cfg   = config["splitter"]
CHUNK_SIZE      = _splitter_cfg["chunk_size"]
CHUNK_OVERLAP   = _splitter_cfg["chunk_overlap"]
ADD_START_INDEX = _splitter_cfg["add_start_index"]


def split_documents(documents: list) -> list:
    """
    Splits documents into smaller overlapping chunks.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked LangChain Document objects.
    """
    logger.info(
        "Splitting %d document(s) | chunk_size=%d | chunk_overlap=%d | add_start_index=%s",
        len(documents), CHUNK_SIZE, CHUNK_OVERLAP, ADD_START_INDEX,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=ADD_START_INDEX,
    )

    try:
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        logger.error("Failed to split documents: %s", e)
        raise

    logger.info(
        "Splitting complete | input_docs=%d | output_chunks=%d | avg_chunks_per_doc=%.1f",
        len(documents),
        len(chunks),
        len(chunks) / len(documents) if documents else 0,
    )
    return chunks
