"""
main.py — CLI entry point for the RAG pipeline.
Loads documents → chunks → vectorstore → RAG chain → interactive Q&A loop.
"""

import os
from dotenv import load_dotenv

from src.loader import load_documents
from src.processor import split_documents
from src.database import create_vector_store
from src.chain import create_qa_chain
from src.config_loader import config
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("DocMind RAG — CLI mode starting up")
    logger.info("=" * 60)

    data_dir = config["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    logger.debug("Data directory ensured: '%s'", data_dir)

    # ── Load ─────────────────────────────────────────────────────
    logger.info("Step 1/4 — Loading documents from '%s'...", data_dir)
    raw_docs = load_documents(data_dir)

    if not raw_docs:
        logger.warning(
            "No documents found in '%s'. "
            "Please add at least one .txt, .docx, .csv, or .pdf file.",
            data_dir,
        )
        print(
            f"\n⚠  No documents found in '{data_dir}'.\n"
            "   Please add at least one .txt, .docx, .csv, or .pdf file."
        )
        return

    logger.info("Loaded %d raw document(s).", len(raw_docs))

    # ── Chunk ─────────────────────────────────────────────────────
    logger.info("Step 2/4 — Chunking documents...")
    chunks = split_documents(raw_docs)
    logger.info("Created %d chunk(s).", len(chunks))

    # ── Vectorstore ───────────────────────────────────────────────
    logger.info("Step 3/4 — Initialising FAISS HNSW vector store...")
    vector_db = create_vector_store(chunks)
    logger.info("Vector store ready.")

    # ── Chain ─────────────────────────────────────────────────────
    logger.info("Step 4/4 — Building RAG chain...")
    rag_chain = create_qa_chain(vector_db)
    logger.info("RAG chain ready. Entering interactive Q&A loop.")

    # ── Interactive Q&A loop ──────────────────────────────────────
    print("\n✅ DocMind is ready! Type 'exit' to quit.\n")
    while True:
        try:
            user_query = input("Ask a question about your docs (or type 'exit'): ").strip()

            if user_query.lower() == "exit":
                logger.info("User exited the CLI loop.")
                print("\nGoodbye! 👋")
                break

            if not user_query:
                continue

            logger.info("User query received | length=%d", len(user_query))
            response = rag_chain.invoke(user_query)
            logger.info("Response generated | length=%d", len(response))
            print(f"\n🤖 AI: {response}\n")

        except KeyboardInterrupt:
            logger.info("CLI session interrupted by user (KeyboardInterrupt).")
            print("\n\nInterrupted. Goodbye! 👋")
            break

        except Exception as e:
            logger.error("Error during query processing: %s", e, exc_info=True)
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
