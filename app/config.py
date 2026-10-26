
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medical AI Decision Support Prototype"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-3.5-lightning:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # RAG
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

    # X-ray prediction
    model_weights_path: Path = Path(
        "artifacts/DeepLearning_Files/"
        "densenet121_cross_attention.weights.h5"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
