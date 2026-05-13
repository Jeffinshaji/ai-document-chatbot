

# ********HNSW for large dataset use this code ***********

import os
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    
    # 1. Determine dimension size
    sample_text = "Sample text to get dimensions."
    dimension = len(embeddings.embed_query(sample_text))
    
    # 2. Configure HNSW index
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_L2)
    
    # 3. Initialize FAISS with HNSW index
    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # 4. Add documents in bulk
    print("💎 Generating embeddings and indexing with HNSW...")
    vectorstore.add_documents(documents)
    
    return vectorstore



