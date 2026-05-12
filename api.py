import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.loader import load_documents
from src.processor import split_documents
from src.database import create_vector_store
from src.chain import create_qa_chain

# PDF support patch — add to loader if not present
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ────────────────────────────────────────────────
rag_chain = None
uploaded_files: list[str] = []

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".csv", ".docx", ".pdf"}


# ── Patched loader that also handles PDF ────────────────────────
def load_documents_with_pdf(directory_path: str):
    documents = []
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".txt":
                from langchain_community.document_loaders import TextLoader
                documents.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == ".docx":
                from langchain_community.document_loaders import Docx2txtLoader
                documents.extend(Docx2txtLoader(path).load())
            elif ext == ".csv":
                from langchain_community.document_loaders import CSVLoader
                documents.extend(CSVLoader(path).load())
            elif ext == ".pdf":
                documents.extend(PyPDFLoader(path).load())
        except Exception as e:
            print(f"⚠️ Could not load {file}: {e}")
    return documents


def rebuild_pipeline():
    """Re-index everything in ./data and rebuild the RAG chain."""
    global rag_chain
    raw_docs = load_documents_with_pdf(DATA_DIR)
    if not raw_docs:
        rag_chain = None
        return False
    chunks = split_documents(raw_docs)
    vector_db = create_vector_store(chunks)
    rag_chain = create_qa_chain(vector_db)
    return True


# ── Models ──────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ── Routes ──────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Accept one or more files, save to ./data, rebuild the pipeline."""
    global uploaded_files
    saved = []
    rejected = []

    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(upload.filename)
            continue
        dest = os.path.join(DATA_DIR, upload.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(upload.filename)
        if upload.filename not in uploaded_files:
            uploaded_files.append(upload.filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid files uploaded.")

    # Rebuild vector store + chain
    ok = rebuild_pipeline()
    return {
        "status": "ready" if ok else "error",
        "saved": saved,
        "rejected": rejected,
        "total_files": len(uploaded_files),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Answer a question using the RAG chain."""
    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded yet. Please upload files first.",
        )
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = rag_chain.invoke(req.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def list_files():
    """Return the list of currently loaded files."""
    files = [f for f in os.listdir(DATA_DIR)
             if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
    return {"files": files}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Remove a file from ./data and rebuild the pipeline."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    os.remove(path)
    if filename in uploaded_files:
        uploaded_files.remove(filename)
    ok = rebuild_pipeline()
    return {"status": "ready" if ok else "empty", "deleted": filename}


@app.get("/status")
def status():
    return {
        "ready": rag_chain is not None,
        "files": len([f for f in os.listdir(DATA_DIR)
                      if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]),
    }
