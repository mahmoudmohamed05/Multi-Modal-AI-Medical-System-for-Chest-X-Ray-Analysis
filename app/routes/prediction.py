from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.ml.predictor import (InvalidImageError, ModelNotAvailableError, PneumoniaPredictor,
                              get_predictor)
from app.rag.context import format_prediction_context
from app.rag.chain import answer_question
from app.schemas import ExplainResponse, PredictionResult

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}


async def _read_image(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG chest X-ray.")
    limit = settings.max_image_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_image_mb} MB.")
    return data


def _predict(predictor: PneumoniaPredictor, data: bytes) -> dict:
    try:
        return predictor.predict(data)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray (PNG/JPEG)"),
    predictor: PneumoniaPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Classify a chest X-ray as NORMAL / PNEUMONIA with the notebook's best model."""
    data = await _read_image(file, settings)
    return await run_in_threadpool(_predict, predictor, data)


@router.post("/explain", response_model=ExplainResponse)
async def predict_and_explain(
    file: UploadFile = File(...),
    question: str = Form("What does the retrieved literature say about the clinical evaluation of suspected pneumonia?",
                         min_length=3, max_length=2000),
    clinical_context: str = Form("", max_length=4000),
    predictor: PneumoniaPredictor = Depends(get_predictor),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Classify the image, then answer `question` with the RAG chatbot using the score as context."""
    data = await _read_image(file, settings)
    prediction = await run_in_threadpool(_predict, predictor, data)
    context = format_prediction_context(prediction, clinical_context)
    try:
        chat = await run_in_threadpool(answer_question, question, context)
    except RuntimeError as exc:
        msg = str(exc)
        raise HTTPException(status_code=409 if msg.startswith("No indexed evidence") else 503, detail=msg)
    return {"prediction": prediction, "chat": chat}
