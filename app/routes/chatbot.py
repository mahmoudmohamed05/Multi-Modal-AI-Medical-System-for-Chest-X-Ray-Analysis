import re
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.rag.bm25_store import load_corpus

from app.rag.context import format_prediction_context
from app.rag.document_loader import SUPPORTED_EXTENSIONS
from app.rag.chain import answer_question
from app.schemas import ChatRequest, ChatResponse, IngestResponse, UploadResponse



router = APIRouter(prefix="/api", tags=["chatbot"])


def require_admin(x_api_key: str = Header(default=""), settings: Settings = Depends(get_settings)) -> None:
    """If ADMIN_API_KEY is set, document management requires a matching X-API-Key header."""
    if settings.admin_api_key and not secrets.compare_digest(x_api_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    """Answer a clinician's question using only retrieved, cited passages."""
    context = format_prediction_context(
        req.prediction.model_dump() if req.prediction else None, req.clinical_context)
    try:
        return answer_question(req.question, context, req.top_k)
    except RuntimeError as exc:
        msg = str(exc)
        # No evidence indexed yet -> 409 (fixable by ingesting); LLM/config problems -> 503.
        status = 409 if msg.startswith("No indexed evidence") else 503
        raise HTTPException(status_code=status, detail=msg)


def _safe_name(name: str) -> str:
    name = Path(name or "").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name or "document"


@router.post("/documents/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
def ingest() -> dict:
    """(Re)build the vector + BM25 indexes from data/medical_documents."""
    from app.rag.ingest import ingest_directory   # lazy: pulls in torch / chromadb
    try:
        return {"chunks_indexed": ingest_directory()}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/documents/upload", response_model=UploadResponse, dependencies=[Depends(require_admin)])
async def upload_document(
    file: UploadFile = File(...),
    reindex: bool = True,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Upload an approved PDF/TXT/MD reference document and (optionally) re-index."""
    name = _safe_name(file.filename)
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Only {sorted(SUPPORTED_EXTENSIONS)} files are accepted.")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    (settings.documents_dir / name).write_bytes(data)

    result = {"filename": name, "bytes_saved": len(data), "indexed": False, "chunks_indexed": None}
    if reindex:
        from app.rag.ingest import ingest_directory
        try:
            result["chunks_indexed"] = await run_in_threadpool(ingest_directory)
            result["indexed"] = True
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return result


def indexed_chunk_count() -> int:
    return len(load_corpus())
