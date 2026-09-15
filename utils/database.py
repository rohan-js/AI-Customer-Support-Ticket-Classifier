"""
SQLite Database Module
=======================
Lightweight persistence layer for logging API predictions.

Why SQLite?
- Zero configuration, single file, no server needed
- Built into Python standard library
- Perfect for local PoC and development
- File-based, easy to inspect, backup, and migrate

Tables:
    predictions: Logs every prediction made through the API
"""

import os
import sqlite3
from typing import List, Dict

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "predictions.db",
)


def get_connection(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Initialize the database schema. Safe to call multiple times."""
    target = db_path or DB_PATH
    os.makedirs(os.path.dirname(target), exist_ok=True)
    conn = get_connection(target)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            priority TEXT NOT NULL,
            reply TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized -> {DB_PATH}")


def log_prediction(text, category, priority, reply, timestamp, db_path=None):
    """Log a prediction to the database."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (text, category, priority, reply, timestamp) VALUES (?, ?, ?, ?, ?)",
        (text, category, priority, reply, timestamp),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=20, db_path=None):
    """Retrieve the most recent predictions."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, text, category, priority, timestamp FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_analytics_summary(db_path=None):
    """Get aggregate analytics from logged predictions."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT category, COUNT(*) as count FROM predictions GROUP BY category ORDER BY count DESC")
    by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("SELECT priority, COUNT(*) as count FROM predictions GROUP BY priority ORDER BY count DESC")
    by_priority = {row["priority"]: row["count"] for row in cursor.fetchall()}

    conn.close()
    return {"total": total, "by_category": by_category, "by_priority": by_priority}


if __name__ == "__main__":
    init_db()
    log_prediction(
        text="Test ticket for billing issue",
        category="Billing Issue",
        priority="High",
        reply="Thank you for contacting us...",
        timestamp="2025-07-01T12:00:00",
    )
    print(f"\nRecent predictions: {get_recent_predictions(5)}")
    print(f"Analytics summary: {get_analytics_summary()}")
