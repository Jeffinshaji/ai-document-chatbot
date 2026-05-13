
# *********** HNSW for large dataset use this code ***********
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """
    Splits documents into smaller chunks.
    chunk_size: Max number of characters per chunk.
    chunk_overlap: Preserves context between chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks
