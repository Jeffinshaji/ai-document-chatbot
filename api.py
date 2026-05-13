"""
api.py — FastAPI backend for the RAG pipeline.
Handles file uploads, document ingestion, chat, and file management.
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.loader import load_documents
from src.processor import split_documents
from src.database import create_vector_store, get_available_sources
from src.chain import create_qa_chain
from src.config_loader import config
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# ── Config values ────────────────────────────────────────────────
_api_cfg    = config["api"]
_path_cfg   = config["paths"]
_loader_cfg = config["loader"]
_app_cfg    = config["app"]

DATA_DIR             = _path_cfg["data_dir"]
FAISS_INDEX_PATH     = _path_cfg["faiss_index"]
ALLOWED_EXTENSIONS   = set(_loader_cfg["supported_extensions"])
CORS_ORIGINS         = _api_cfg["cors_origins"]
API_TITLE            = _api_cfg["title"]

# ── App setup ────────────────────────────────────────────────────
app = FastAPI(title=API_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI app initialised | title='%s'", API_TITLE)

# ── Global state ─────────────────────────────────────────────────
vector_db = None   # kept alive for metadata filtering
rag_chain  = None  # default chain (no filter)

os.makedirs(DATA_DIR, exist_ok=True)
logger.debug("Data directory ensured: '%s'", DATA_DIR)


# ── Pipeline builder ─────────────────────────────────────────────
def rebuild_pipeline() -> bool:
    """
    Re-indexes all files in DATA_DIR and rebuilds the RAG chain.

    Returns:
        True if pipeline is ready, False if no documents found.
    """
    global vector_db, rag_chain

    logger.info("Rebuilding pipeline from directory: '%s'", DATA_DIR)
    raw_docs = load_documents(DATA_DIR)

    if not raw_docs:
        logger.warning("No documents found in '%s'. Pipeline not built.", DATA_DIR)
        vector_db = None
        rag_chain  = None
        return False

    chunks    = split_documents(raw_docs)
    vector_db = create_vector_store(chunks)
    rag_chain  = create_qa_chain(vector_db)
    logger.info("Pipeline rebuild complete. RAG chain is ready.")
    return True


# ── Request / Response models ────────────────────────────────────
class ChatRequest(BaseModel):
    question      : str
    source_filter : str | None = None


class ChatResponse(BaseModel):
    answer  : str
    sources : list[str] = []


# ── Routes ───────────────────────────────────────────────────────

@app.get("/")
def root():
    logger.debug("GET / — serving index.html")
    return FileResponse("index.html")


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Accept files, save to DATA_DIR, rebuild the pipeline."""
    logger.info("POST /upload | file_count=%d", len(files))
    saved    = []
    rejected = []

    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning("Rejected unsupported file: '%s' (ext=%s)", upload.filename, ext)
            rejected.append(upload.filename)
            continue

        dest = os.path.join(DATA_DIR, upload.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(upload.filename)
        logger.info("Saved uploaded file: '%s'", upload.filename)

    if not saved:
        logger.warning("No valid files in upload batch. Rejected: %s", rejected)
        raise HTTPException(status_code=400, detail="No valid files uploaded.")

    logger.info("Triggering pipeline rebuild after upload | saved=%s", saved)
    ok = rebuild_pipeline()

    return {
        "status"      : "ready" if ok else "error",
        "saved"       : saved,
        "rejected"    : rejected,
        "total_files" : len(saved),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Answer a question using the RAG chain.
    Optionally filter by a specific source file.
    """
    logger.info(
        "POST /chat | question_len=%d | source_filter=%s",
        len(req.question), req.source_filter or "none",
    )

    if vector_db is None:
        logger.warning("Chat attempted but no documents are loaded.")
        raise HTTPException(
            status_code=400,
            detail="No documents loaded yet. Please upload files first.",
        )

    if not req.question.strip():
        logger.warning("Empty question received.")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        if req.source_filter:
            logger.debug("Using filtered chain for source: '%s'", req.source_filter)
            chain = create_qa_chain(vector_db, source_filter=req.source_filter)
        else:
            chain = rag_chain

        answer  = chain.invoke(req.question)
        sources = get_available_sources(vector_db) if vector_db else []
        logger.info("Chat response generated | sources_available=%d", len(sources))

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error("Error during chat inference: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def list_files():
    """Return list of currently loaded files."""
    files = [
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    logger.debug("GET /files | count=%d", len(files))
    return {"files": files}


@app.get("/sources")
def list_sources():
    """Return unique source filenames available for filtering."""
    if vector_db is None:
        logger.debug("GET /sources — no vectorstore loaded.")
        return {"sources": []}
    sources = get_available_sources(vector_db)
    logger.debug("GET /sources | count=%d", len(sources))
    return {"sources": sources}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Remove a file and rebuild the pipeline."""
    logger.info("DELETE /files/%s", filename)
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        logger.warning("Delete failed — file not found: '%s'", filename)
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(path)
    logger.info("File deleted: '%s'", filename)

    # Clear saved FAISS index so it rebuilds cleanly
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        logger.info("FAISS index cleared for clean rebuild after deletion.")

    ok = rebuild_pipeline()
    logger.info("Pipeline status after deletion: %s", "ready" if ok else "empty")
    return {"status": "ready" if ok else "empty", "deleted": filename}


@app.get("/status")
def status():
    files = [
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
    ]
    logger.debug("GET /status | ready=%s | files=%d", rag_chain is not None, len(files))
    return {
        "ready"   : rag_chain is not None,
        "files"   : len(files),
        "sources" : get_available_sources(vector_db) if vector_db else [],
    }
