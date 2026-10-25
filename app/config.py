from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical AI Decision Support Prototype"
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemini-2.0-flash-001"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "medical_documents"
    documents_dir: Path = Path("data/medical_documents")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    vector_top_k: int = 8
    bm25_top_k: int = 8
    final_top_k: int = 6
    rrf_k: int = 60
    max_upload_mb: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
