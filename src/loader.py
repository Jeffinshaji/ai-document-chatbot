# **********use HNSW for large dataset use this code ***********

import os
from langchain_community.document_loaders import TextLoader, CSVLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader

def load_documents(directory_path):
    documents = []
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        ext  = os.path.splitext(file)[1].lower()
        try:
            if ext == ".txt":
                docs = TextLoader(path, encoding="utf-8").load()
                for doc in docs:
                    lines = doc.page_content.split("\n")
                    doc.metadata["total_lines"] = len(lines)
                    doc.metadata["file_type"]   = "txt"
                documents.extend(docs)

            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
                for doc in docs:
                    paragraphs = [p for p in doc.page_content.split("\n") if p.strip()]
                    doc.metadata["total_paragraphs"] = len(paragraphs)
                    doc.metadata["file_type"]        = "docx"
                documents.extend(docs)

            elif ext == ".csv":
                docs = CSVLoader(path).load()
                for i, doc in enumerate(docs):
                    doc.metadata["row"]       = i + 1
                    doc.metadata["file_type"] = "csv"
                documents.extend(docs)

            elif ext == ".pdf":
                docs = PyPDFLoader(path).load()
                for doc in docs:
                    doc.metadata["file_type"] = "pdf"
                documents.extend(docs)

        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")

    return documents