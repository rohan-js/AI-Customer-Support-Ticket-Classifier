"""Tests for the SQLite prediction-logging layer."""

from utils.database import (
    get_analytics_summary,
    get_recent_predictions,
    init_db,
    log_prediction,
)


def _db(tmp_path):
    return str(tmp_path / "predictions.db")


def test_roundtrip_log_and_read(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    log_prediction(
        text="I was charged twice",
        category="Billing Issue",
        priority="High",
        reply="Thank you...",
        timestamp="2026-01-01T00:00:00",
        db_path=db,
    )
    rows = get_recent_predictions(10, db_path=db)
    assert len(rows) == 1
    assert rows[0]["category"] == "Billing Issue"
    assert rows[0]["priority"] == "High"
    assert rows[0]["timestamp"] == "2026-01-01T00:00:00"


def test_init_is_idempotent(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    init_db(db)  # must not raise
    assert get_recent_predictions(5, db_path=db) == []


def test_recent_predictions_limit_and_order(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    for i in range(5):
        log_prediction(f"t{i}", "General Inquiry", "Low", "r", f"2026-01-0{i + 1}T00:00:00", db_path=db)
    rows = get_recent_predictions(2, db_path=db)
    assert [r["text"] for r in rows] == ["t4", "t3"]  # newest first


def test_analytics_summary(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    log_prediction("a", "Billing Issue", "High", "r", "2026-01-01T00:00:00", db_path=db)
    log_prediction("b", "Billing Issue", "Low", "r", "2026-01-02T00:00:00", db_path=db)
    log_prediction("c", "Feature Request", "Low", "r", "2026-01-03T00:00:00", db_path=db)
    summary = get_analytics_summary(db_path=db)
    assert summary["total"] == 3
    assert summary["by_category"]["Billing Issue"] == 2
    assert summary["by_priority"]["Low"] == 2
