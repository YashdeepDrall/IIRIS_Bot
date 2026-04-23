import os
from datetime import datetime
from typing import Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo import MongoClient


class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""

    question: str
    k: int = 15
    temperature: float = 0.7
    max_tokens: int = 512
    user_id: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Request model for the /feedback endpoint."""

    chat_id: str
    feedback: str


class ChatRecord(BaseModel):
    """Database schema for storing a chat interaction."""

    question: str
    answer: str
    context: str
    flags: List[str] = Field(default_factory=list)
    user_feedback: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


load_dotenv()


def get_mongo_collection():
    """Establishes connection to MongoDB and returns the chat_history collection."""

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("MONGO_URI not found. Database features will be disabled.")
        return None

    client = MongoClient(mongo_uri)
    db = client["cybersecurity_bot"]
    return db["chat_history"]


def store_chat(
    question: str,
    answer: str,
    context: str,
    flags: Optional[List[str]] = None,
    token_usage: Optional[Dict[str, int]] = None,
) -> str:
    """Stores the chat interaction in MongoDB."""

    try:
        if flags is None:
            flags = []

        collection = get_mongo_collection()
        if collection is None:
            return "local_mode"

        record = ChatRecord(
            question=question,
            answer=answer,
            context=context,
            flags=flags,
            token_usage=token_usage,
        )
        result = collection.insert_one(record.model_dump())
        print(f"Chat stored in MongoDB with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as error:
        print(f"Error storing chat in MongoDB: {error}")
        return ""


def update_chat_feedback(chat_id: str, feedback: str):
    """Updates a chat record with user feedback."""

    try:
        collection = get_mongo_collection()
        if collection is None:
            return

        collection.update_one({"_id": ObjectId(chat_id)}, {"$set": {"user_feedback": feedback}})
        print(f"Feedback updated for {chat_id}: {feedback}")
    except Exception as error:
        print(f"Error updating feedback: {error}")
