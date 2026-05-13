from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ])

def rerank_docs(query, docs, top_n=3):
    """
    Manual reranker — scores chunks by
    how many query words appear in the chunk.
    No external library needed.
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        content_words = set(doc.page_content.lower().split())
        score = len(query_words & content_words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

def create_qa_chain(vectorstore, source_filter=None):
    # ── 1. LLM ──────────────────────────────────────────────
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # ── 2. Search kwargs ─────────────────────────────────────
    search_kwargs = {"k": 10}
    if source_filter:
        search_kwargs["filter"] = {"source": source_filter}

    # ── 3. Base retriever ────────────────────────────────────
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # ── 4. Prompt ────────────────────────────────────────────
    template = """Answer the question based only on the following context.
Each source is labeled — mention the source when relevant.

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # ── 5. RAG chain with manual reranking ───────────────────
    def retrieve_and_rerank(question):
        docs = retriever.invoke(question)
        reranked = rerank_docs(question, docs, top_n=3)
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

    return rag_chain