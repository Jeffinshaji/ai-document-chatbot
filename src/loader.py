# import os
# from langchain_community.document_loaders import TextLoader, CSVLoader, Docx2txtLoader

# def load_documents(directory_path):
#     documents = []
#     for file in os.listdir(directory_path):
#         path = os.path.join(directory_path, file)
#         if file.endswith('.txt'):
#             documents.extend(TextLoader(path).load())
#         elif file.endswith('.docx'):
#             documents.extend(Docx2txtLoader(path).load())
#         elif file.endswith('.csv'):
#             documents.extend(CSVLoader(path).load())
#     return documents

# **********use HNSW for large dataset use this code ***********

import os
from langchain_community.document_loaders import TextLoader, CSVLoader, Docx2txtLoader

def load_documents(directory_path):
    documents = []
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        if file.endswith('.txt'):
            documents.extend(TextLoader(path, encoding='utf-8').load())
        elif file.endswith('.docx'):
            documents.extend(Docx2txtLoader(path).load())
        elif file.endswith('.csv'):
            documents.extend(CSVLoader(path).load())
    return documents