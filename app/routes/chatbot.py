
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.chain import answer_question, ChatbotError

router = APIRouter(prefix="/api/chat", tags=["chatbot"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    clinical_context: str = Field(default="", max_length=4000)


@router.get("/health")
def chat_health():
    return {"status": "ok", "service": "medical-chatbot"}


@router.post("")
def chat(request: ChatRequest):
    try:
        return answer_question(
            question=request.question,
            clinical_context=request.clinical_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ChatbotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
