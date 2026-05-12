# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# def create_vector_store(documents):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(documents, embeddings)
#     return vectorstore

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

# ********* for large datasets use this code 1 **************

# import os
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# # from langchain_core.retrievers import BaseRetriever
# # from langchain.retrievers import EnsembleRetriever



# def create_vector_store(documents):
#     embeddings = OpenAIEmbeddings()
#     index_path = "faiss_index"

#     # 1. Check if index already exists to save time/money
#     if os.path.exists(index_path):
#         print("📁 Loading existing FAISS index...")
#         vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         print("💎 Generating embeddings (this may take time for large files)...")
#         vectorstore = FAISS.from_documents(documents, embeddings)
#         vectorstore.save_local(index_path)

#     # 2. Setup Hybrid Search: Combine FAISS (Semantic) with BM25 (Keyword)
#     # This is much faster and more accurate for large haystacks
#     bm25_retriever = BM25Retriever.from_documents(documents)
#     bm25_retriever.k = 3 
    
#     faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # Combine them: 70% Vector, 30% Keyword
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[bm25_retriever, faiss_retriever], 
#         weights=[0.3, 0.7]
#     )
    
#     return ensemble_retriever

# ***************** for large datasets use this code 2 (with caching and more efficient retrieval)**************

# import os
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS

# def create_vector_store(documents):
#     embeddings = OpenAIEmbeddings()
#     index_path = "faiss_index"

#     # Persistence: Saves you from re-embedding 10,000 pages every time!
#     if os.path.exists(index_path):
#         print("📁 Loading existing FAISS index...")
#         vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         print("💎 Generating embeddings (this will take time for 10,000 pages)...")
#         vectorstore = FAISS.from_documents(documents, embeddings)
#         vectorstore.save_local(index_path)

#     # We try to use Hybrid Search, but fall back if the library is missing
#     try:
#         from langchain_community.retrievers import BM25Retriever
#         from langchain.retrievers import EnsembleRetriever
        
#         print("🔍 Using Hybrid Search (FAISS + BM25)")
#         bm25_retriever = BM25Retriever.from_documents(documents)
#         faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
#         return EnsembleRetriever(
#             retrievers=[bm25_retriever, faiss_retriever], 
#             weights=[0.3, 0.7]
#         )
#     except ImportError:
#         print("⚠️ Advanced retrievers not found. Using standard FAISS search.")
#         return vectorstore.as_retriever(search_kwargs={"k": 5})