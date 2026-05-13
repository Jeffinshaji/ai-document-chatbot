"""
loader.py — Loads documents from a directory.
Supports: .txt, .docx, .csv, .pdf
Enriches each document's metadata with file-type-specific info.
"""

import os
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
)

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)

# ── Config values ────────────────────────────────────────────────
_loader_cfg          = config["loader"]
ENCODING             = _loader_cfg["encoding"]
SUPPORTED_EXTENSIONS = set(_loader_cfg["supported_extensions"])


def load_documents(directory_path: str) -> list:
    """
    Loads all supported documents from a directory.

    Args:
        directory_path: Path to the folder containing documents.

    Returns:
        List of LangChain Document objects with enriched metadata.
    """
    logger.info("Loading documents from directory: '%s'", directory_path)

    if not os.path.exists(directory_path):
        logger.warning("Directory does not exist: '%s'", directory_path)
        return []

    all_files = os.listdir(directory_path)
    logger.debug("Files found in directory: %s", all_files)

    documents = []

    for file in all_files:
        path = os.path.join(directory_path, file)
        ext  = os.path.splitext(file)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file type: '%s' (ext=%s)", file, ext)
            continue

        try:
            if ext == ".txt":
                docs = TextLoader(path, encoding=ENCODING).load()
                for doc in docs:
                    lines = doc.page_content.split("\n")
                    doc.metadata["total_lines"] = len(lines)
                    doc.metadata["file_type"]   = "txt"
                logger.info("Loaded TXT  | file='%s' | pages/docs=%d", file, len(docs))
                documents.extend(docs)

            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
                for doc in docs:
                    paragraphs = [p for p in doc.page_content.split("\n") if p.strip()]
                    doc.metadata["total_paragraphs"] = len(paragraphs)
                    doc.metadata["file_type"]        = "docx"
                logger.info("Loaded DOCX | file='%s' | docs=%d", file, len(docs))
                documents.extend(docs)

            elif ext == ".csv":
                docs = CSVLoader(path).load()
                for i, doc in enumerate(docs):
                    doc.metadata["row"]       = i + 1
                    doc.metadata["file_type"] = "csv"
                logger.info("Loaded CSV  | file='%s' | rows=%d", file, len(docs))
                documents.extend(docs)

            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
                for doc in docs:
                    doc.metadata["file_type"] = "pdf"
                logger.info("Loaded PDF  | file='%s' | pages=%d", file, len(docs))
                documents.extend(docs)

        except Exception as e:
            logger.warning("Could not load file '%s': %s", file, e)

    logger.info(
        "Document loading complete | total_files_loaded=%d | total_documents=%d",
        sum(1 for f in all_files if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS),
        len(documents),
    )
    return documents
