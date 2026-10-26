
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.rag.chain import answer_question, ChatbotError
from app.services.prediction_service import prediction_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prediction", tags=["prediction"])

MAX_IMAGE_BYTES = 10 * 1024 * 1024


async def read_image(file: UploadFile) -> bytes:
    content_type = (file.content_type or "").lower()

    if content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG or PNG image.",
        )

    data = await file.read(MAX_IMAGE_BYTES + 1)

    if not data:
        raise HTTPException(
            status_code=422,
            detail="Image is empty.",
        )

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB.",
        )

    return data


def run_prediction(data: bytes) -> dict:
    try:
        return prediction_service.predict(data)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        logger.exception("Model weights file not found")

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Image prediction failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="Image prediction failed. Check server logs.",
        ) from exc


@router.post("")
async def predict_image(file: UploadFile = File(...)):
    data = await read_image(file)

    return await run_in_threadpool(
        run_prediction,
        data,
    )


@router.post("/explain")
async def predict_and_answer(
    file: UploadFile = File(...),
    question: str = Form(
        "Explain what the retrieved literature says about pneumonia evaluation.",
        min_length=3,
        max_length=2000,
    ),
):
    data = await read_image(file)

    prediction = await run_in_threadpool(
        run_prediction,
        data,
    )

    context = (
        "Automated research-model output, not a confirmed diagnosis: "
        f"class={prediction['label']}; "
        "estimated pneumonia probability="
        f"{prediction['probability_pneumonia']:.4f}. "
        "Use this only as uncertain context and do not treat it as clinical proof."
    )

    try:
        answer = await run_in_threadpool(
            answer_question,
            question,
            context,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except ChatbotError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "prediction": prediction,
        "chat": answer,
    }
