from typing import Literal
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    label: Literal["NORMAL", "PNEUMONIA"]
    probability_pneumonia: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted label")
    threshold: float
    model_name: str
    disclaimer: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    clinical_context: str = Field(default="", max_length=4000)
    top_k: int = Field(default=6, ge=1, le=15)
    prediction: PredictionResult | None = Field(
        default=None,
        description="Optional output of /api/predictions; added to the clinical context for the LLM.",
    )


class SourceItem(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    disclaimer: str


class ExplainResponse(BaseModel):
    prediction: PredictionResult
    chat: ChatResponse


class IngestResponse(BaseModel):
    chunks_indexed: int


class UploadResponse(BaseModel):
    filename: str
    bytes_saved: int
    indexed: bool
    chunks_indexed: int | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    classifier_weights_present: bool
    classifier_loaded: bool
    indexed_chunks_bm25: int
    llm_configured: bool
