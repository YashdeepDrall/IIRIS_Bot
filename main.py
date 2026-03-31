import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import FeedbackRequest, QueryRequest, update_chat_feedback
from services import get_rag_system


load_dotenv()


def get_allowed_origins():
    raw_value = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
    if raw_value == "*" or not raw_value:
        return ["*"]
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


app = FastAPI(title="IIRIS Gemini RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "running", "message": "IIRIS Gemini RAG API is active"}


@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        rag = get_rag_system()
        return await rag.answer(request)
    except Exception as error:
        print(f"Ask route failed: {repr(error)}")
        traceback.print_exc()
        return {
            "answer": "The assistant is temporarily unavailable. Please try again in a moment.",
            "context": "",
            "docs": [],
            "chat_id": "",
        }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    update_chat_feedback(request.chat_id, request.feedback)
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    print("Starting local Gemini RAG server...")
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 10000)))
