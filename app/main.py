import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.ml.predictor import ModelNotAvailableError, get_predictor
from app.routes import api_router
from app.routes.chatbot import indexed_chunk_count
from app.schemas import HealthResponse

log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.preload_model:
        try:
            get_predictor().load()
            log.info("Classifier loaded from %s", settings.model_weights_path)
        except ModelNotAvailableError as exc:
            log.warning("Classifier not preloaded: %s", exc)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    from pathlib import Path
    from fastapi.responses import FileResponse

    _UI = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(_UI)

    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins,
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> dict:
        predictor = get_predictor()
        key = settings.openrouter_api_key
        return {
            "status": "ok",
            "classifier_weights_present": predictor.weights_present,
            "classifier_loaded": predictor.is_loaded,
            "indexed_chunks_bm25": indexed_chunk_count(),
            "llm_configured": bool(key and key != "replace_me"),
        }

    return app


app = create_app()
