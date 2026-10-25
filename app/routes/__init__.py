from fastapi import APIRouter

from app.routes import chatbot, prediction

api_router = APIRouter()
api_router.include_router(chatbot.router)
api_router.include_router(prediction.router)

__all__ = ["api_router"]