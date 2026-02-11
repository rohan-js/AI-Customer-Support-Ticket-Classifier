"""
FastAPI REST Service for Ticket Classification
================================================
Endpoints:
    POST /predict   - Classify ticket, predict priority, generate reply
    GET  /health    - Health check with model status
    GET  /analytics - Aggregate analytics from prediction history
    GET  /history   - Recent prediction log
"""

import os
import sys
from datetime import datetime
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import clean_text
from utils.reply_generator import generate_reply
from utils.database import init_db, log_prediction, get_recent_predictions, get_analytics_summary


# Global model store
models = {}


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Ticket text to classify")


class PredictionResponse(BaseModel):
    category: str
    priority: str
    reply: str
    timestamp: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    saved_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "saved"
    )

    # Load category model
    try:
        models["category_model"] = joblib.load(
            os.path.join(saved_dir, "category_baseline_model.joblib")
        )
        models["category_tfidf"] = joblib.load(
            os.path.join(saved_dir, "category_tfidf_vectorizer.joblib")
        )
        models["category_encoder"] = joblib.load(
            os.path.join(saved_dir, "category_label_encoder.joblib")
        )
        print("[OK] Category model loaded (TF-IDF + Logistic Regression)")
    except FileNotFoundError:
        print("[X] Category model not found -- run 'python models/train_baseline.py' first")

    # Load priority model
    try:
        models["priority_model"] = joblib.load(
            os.path.join(saved_dir, "priority_rf_model.joblib")
        )
        models["priority_tfidf"] = joblib.load(
            os.path.join(saved_dir, "priority_tfidf_vectorizer.joblib")
        )
        models["priority_encoder"] = joblib.load(
            os.path.join(saved_dir, "priority_label_encoder.joblib")
        )
        print("[OK] Priority model loaded (TF-IDF + Random Forest)")
    except FileNotFoundError:
        print("[X] Priority model not found -- run 'python models/train_priority.py' first")

    # Initialize SQLite database
    init_db()
    print("[OK] SQLite database initialized")

    yield  # App runs here

    models.clear()
    print("[OK] Models unloaded")


app = FastAPI(
    title="AI Customer Support Ticket Classifier",
    description="Classifies support tickets, predicts priority, and generates auto-replies",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(ticket: TicketRequest):
    """Classify a support ticket and generate an auto-reply."""
    cleaned = clean_text(ticket.text)
    timestamp = datetime.now().isoformat()

    # Category prediction
    if "category_model" in models:
        X = models["category_tfidf"].transform([cleaned])
        cat_idx = models["category_model"].predict(X)[0]
        category = models["category_encoder"].inverse_transform([cat_idx])[0]
    else:
        category = "Unknown (model not loaded)"

    # Priority prediction
    if "priority_model" in models:
        X = models["priority_tfidf"].transform([cleaned])
        pri_idx = models["priority_model"].predict(X)[0]
        priority = models["priority_encoder"].inverse_transform([pri_idx])[0]
    else:
        priority = "Medium"

    # Generate reply
    reply = generate_reply(ticket.text, category, priority)

    # Log to database
    log_prediction(
        text=ticket.text,
        category=category,
        priority=priority,
        reply=reply,
        timestamp=timestamp,
    )

    return PredictionResponse(
        category=category,
        priority=priority,
        reply=reply,
        timestamp=timestamp,
    )


@app.get("/health")
async def health():
    """Health check endpoint with model status."""
    return {
        "status": "healthy",
        "models": {
            "category": "category_model" in models,
            "priority": "priority_model" in models,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/analytics")
async def analytics():
    """Get aggregate analytics from prediction history."""
    return get_analytics_summary()


@app.get("/history")
async def history(limit: int = 20):
    """Get recent prediction history."""
    return get_recent_predictions(limit)
