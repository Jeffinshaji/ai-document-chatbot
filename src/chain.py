"""
chain.py — Builds the RAG chain: retriever → reranker → prompt → LLM → parser.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)

# ── Config values ────────────────────────────────────────────────
_llm_cfg       = config["llm"]
_ret_cfg       = config["retriever"]
_rerank_cfg    = config["reranker"]

MODEL_NAME     = _llm_cfg["model_name"]
TEMPERATURE    = _llm_cfg["temperature"]
RETRIEVER_K    = _ret_cfg["k"]
RERANKER_TOP_N = _rerank_cfg["top_n"]


def format_docs(docs):
    """Formats retrieved docs into a labeled context string."""
    logger.debug("Formatting %d retrieved document(s) for context.", len(docs))
    formatted = []
    for doc in docs:
        meta     = doc.metadata
        filetype = meta.get("file_type", "")
        source   = os.path.basename(meta.get("source", "unknown"))

        if filetype == "pdf":
            page     = meta.get("page", None)
            location = f"Page {page + 1}" if page is not None else ""

        elif filetype == "txt":
            start_idx = meta.get("start_index", None)
            location  = f"Around character {start_idx}" if start_idx else ""

        elif filetype == "csv":
            row      = meta.get("row", None)
            location = f"Row {row}" if row else ""

        elif filetype == "docx":
            total_p  = meta.get("total_paragraphs", None)
            location = f"~{total_p} paragraphs total" if total_p else ""

        else:
            location = ""

        label = f"[Source: {source}"
        if location:
            label += f" | {location}"
        label += "]"

        formatted.append(f"{label}\n{doc.page_content}")

    return "\n\n".join(formatted)


def rerank_docs(query, docs, top_n=None):
    """
    Manual keyword reranker — scores chunks by how many query
    words appear in the chunk. No external library needed.

    Args:
        query:  User question string.
        docs:   List of retrieved LangChain Documents.
        top_n:  Number of top docs to keep (defaults to config value).

    Returns:
        List of top_n reranked Documents.
    """
    if top_n is None:
        top_n = RERANKER_TOP_N

    logger.debug(
        "Reranking %d docs → keeping top %d | query: '%s'",
        len(docs), top_n, query[:80],
    )

    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        content_words = set(doc.page_content.lower().split())
        score = len(query_words & content_words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked = [doc for _, doc in scored[:top_n]]
    logger.debug("Reranking complete. Top scores: %s", [s for s, _ in scored[:top_n]])
    return reranked


def create_qa_chain(vectorstore, source_filter=None):
    """
    Builds the full RAG chain.

    Args:
        vectorstore:   FAISS vectorstore instance.
        source_filter: Optional filename string to filter retrieval by source.

    Returns:
        A LangChain Runnable (RAG chain).
    """
    logger.info(
        "Creating QA chain | model=%s | temperature=%s | k=%d | top_n=%d | filter=%s",
        MODEL_NAME, TEMPERATURE, RETRIEVER_K, RERANKER_TOP_N,
        source_filter if source_filter else "none",
    )

    # ── 1. LLM ──────────────────────────────────────────────────
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
    logger.debug("LLM initialised: model=%s, temperature=%s", MODEL_NAME, TEMPERATURE)

    # ── 2. Search kwargs ─────────────────────────────────────────
    search_kwargs = {"k": RETRIEVER_K}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}
        logger.debug("Source filter applied: %s", source_filter)

    # ── 3. Base retriever ────────────────────────────────────────
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    logger.debug("Retriever created with k=%d", RETRIEVER_K)

    # ── 4. Prompt ────────────────────────────────────────────────
    template = """Answer the question based only on the following context.
Each source is labeled — mention the source when relevant.

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # ── 5. RAG chain with manual reranking ───────────────────────
    def retrieve_and_rerank(question):
        logger.debug("Retrieving docs for question: '%s'", question[:80])
        docs = retriever.invoke(question)
        logger.debug("Retrieved %d doc(s) from vectorstore.", len(docs))
        reranked = rerank_docs(question, docs, top_n=RERANKER_TOP_N)
        logger.debug("Reranked to %d doc(s).", len(reranked))
        return format_docs(reranked)

    rag_chain = (
        {
            "context" : RunnableLambda(retrieve_and_rerank),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("QA chain created successfully.")
    return rag_chain
