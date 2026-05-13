import os
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

INDEX_PATH = "./faiss_index"

def create_vector_store(documents):
    """
    Creates or loads a FAISS HNSW vector store.
    - First run  : builds index from scratch and saves to disk
    - Later runs : loads from disk and adds new documents
    """
    embeddings = OpenAIEmbeddings()

    # ── Load existing index from disk if available ───────────────
    if os.path.exists(INDEX_PATH):
        print("⚡ Loading existing FAISS index from disk...")
        vectorstore = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Index loaded! Adding new documents...")
        vectorstore.add_documents(documents)
        vectorstore.save_local(INDEX_PATH)
        print("💾 Index updated and saved!")
        return vectorstore

    # ── First time — build from scratch ─────────────────────────
    print("🔨 Building new FAISS index from scratch...")
    sample_text = "Sample text to get dimensions."
    dimension = len(embeddings.embed_query(sample_text))

    # HNSW index — fast and accurate for small-medium datasets
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_L2)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    print("💎 Generating embeddings and indexing with HNSW...")
    vectorstore.add_documents(documents)

    # Save to disk for future use
    vectorstore.save_local(INDEX_PATH)
    print("💾 Index saved to disk!")

    return vectorstore


def get_available_sources(vectorstore):
    """
    Returns a list of unique source filenames stored in the index.
    Useful for metadata filtering — know what files are available.
    """
    sources = set()
    for doc_id in vectorstore.index_to_docstore_id.values():
        doc = vectorstore.docstore.search(doc_id)
        if doc and hasattr(doc, 'metadata'):
            source = doc.metadata.get('source', '')
            if source:
                # Extract just the filename from full path
                sources.add(os.path.basename(source))
    return sorted(list(sources))
