"""Tests for the FastAPI service (real models loaded via lifespan)."""

import pytest
from fastapi.testclient import TestClient

import api.main as api
import utils.database


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with the prediction DB redirected to a temp file."""
    monkeypatch.setattr(utils.database, "DB_PATH", str(tmp_path / "predictions.db"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["models"]["category"] is True
    assert data["models"]["priority"] is True


def test_predict_happy_path(client):
    resp = client.post(
        "/predict",
        json={"text": "I was charged $49.99 twice for my subscription this month"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "Billing Issue"
    assert data["priority"] in {"Low", "Medium", "High", "Urgent"}
    assert "Dear Customer" in data["reply"] or "Hello" in data["reply"]
    assert data["timestamp"]


def test_predict_too_short_text_rejected(client):
    resp = client.post("/predict", json={"text": "hi"})
    assert resp.status_code == 422


def test_predict_and_history_analytics(client):
    client.post("/predict", json={"text": "Please refund my last payment of $19.99"})
    history = client.get("/history").json()
    assert len(history) == 1
    assert history[0]["category"] == "Refund Request"
    analytics = client.get("/analytics").json()
    assert analytics["total"] == 1
    assert "Refund Request" in analytics["by_category"]


def test_predict_without_models_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.database, "DB_PATH", str(tmp_path / "predictions.db"))

    def _raise_fnf(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(api.joblib, "load", _raise_fnf)
    with TestClient(api.app) as client:
        health = client.get("/health").json()
        assert health["models"]["category"] is False
        resp = client.post("/predict", json={"text": "The app keeps crashing when I upload files"})
        data = resp.json()
        assert data["category"] == "Unknown (model not loaded)"
        assert data["priority"] == "Medium"
        assert len(data["reply"]) > 50  # reply generation still works
