
from fastapi import FastAPI

from app.routes.chatbot import router as chatbot_router
from app.routes.prediction import router as prediction_router

app = FastAPI(title="Medical AI Decision Support")

app.include_router(chatbot_router)
app.include_router(prediction_router)


@app.get("/health")
def health():
    return {"status": "ok"}
