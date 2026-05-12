# import os
# from urllib import response
# from dotenv import load_dotenv
# from src.loader import load_documents
# from src.processor import split_documents
# from src.database import create_vector_store
# from src.chain import create_qa_chain

# load_dotenv()

# def main():
#     # 1. Load
#     print("📂 Loading files...")
#     raw_docs = load_documents("./data")
    
#     # 2. Process
#     print("✂️ Chunking text...")
#     chunks = split_documents(raw_docs)
    
#     # 3. Embed & Store
#     print("💎 Generating embeddings and saving to FAISS...")
#     vector_db = create_vector_store(chunks)
    
#     # 4. Initialize Chain
#     print("🧠 Initializing RAG Chain...")
#     rag_chain = create_qa_chain(vector_db)
    
#     # 5. Interactive Loop
#     while True:
#         user_query = input("\nAsk a question about your docs (or type 'exit'): ")
#         if user_query.lower() == 'exit':
#             break
            
#         #response = rag_chain.invoke({"input": user_query})
#         response = rag_chain.invoke(user_query)
#         print(f"\n🤖 AI: {response}")

# ********HNSW for small dataset use this code ********
import os
from dotenv import load_dotenv
from src.loader import load_documents
from src.processor import split_documents
from src.database import create_vector_store
from src.chain import create_qa_chain

load_dotenv()

def main():
    print("📂 Loading files...")
    os.makedirs("./data", exist_ok=True)
    raw_docs = load_documents("./data")
    
    if not raw_docs:
        print("⚠️ No documents found in ./data folder. Please add at least one .txt, .docx, or .csv file.")
        return
        
    print("✂️ Chunking text...")
    chunks = split_documents(raw_docs)
    
    print("💎 Initializing HNSW FAISS Vector Store...")
    vector_db = create_vector_store(chunks)
    
    print("🧠 Initializing RAG Chain...")
    rag_chain = create_qa_chain(vector_db)
    
    # Interactive Loop
    while True:
        try:
            user_query = input("\nAsk a question about your docs (or type 'exit'): ")
            if user_query.lower() == 'exit':
                break
                
            response = rag_chain.invoke(user_query)
            print(f"\n🤖 AI: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()


# *-*-*-*-*-*-*-* for external knowledge and chat history, use this code -*-*-*-*-*-*

#     # chat_history = []

#     # while True:
#     #     user_query = input("\nAsk a question (or type 'exit'): ")
#     #     if user_query.lower() == 'exit':
#     #         break

#     #     response = rag_chain(user_query, chat_history)

#     #     print(f"\n🤖 AI: {response}")

#     # # store conversation
#     #     chat_history.append(f"User: {user_query}")
#     #     chat_history.append(f"AI: {response}")

# if __name__ == "__main__":
#     main()


# ************ for large dataset use this code ***********
# import os
# from dotenv import load_dotenv
# from src.loader import load_documents
# from src.processor import split_documents
# from src.database import create_vector_store
# from src.chain import create_qa_chain

# load_dotenv()

# def main():
#     # 1. Load
#     print("📂 Loading files...")
#     raw_docs = load_documents("./data")
    
#     # 2. Process
#     print("✂️ Chunking text...")
#     chunks = split_documents(raw_docs)
    
#     # 3. Embed & Store (Now returns the Ensemble Retriever)
#     print("💎 Initializing Retriever (FAISS + BM25)...")
#     # This function now handles checking for existing indexes automatically
#     retriever = create_vector_store(chunks)
    
#     # 4. Initialize Chain
#     print("🧠 Initializing RAG Chain...")
#     # Pass the retriever directly
#     rag_chain = create_qa_chain(retriever)
    
#     chat_history = []

#     print("\n✅ System Ready! (Using Hybrid Search)")
#     while True:
#         user_query = input("\nAsk a question (or type 'exit'): ")
#         if user_query.lower() == 'exit':
#             break

#         response = rag_chain(user_query, chat_history)

#         print(f"\n🤖 AI: {response}")

#         chat_history.append(f"User: {user_query}")
#         chat_history.append(f"AI: {response}")

# if __name__ == "__main__":
#     main()