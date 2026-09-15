"""Tests for the synthetic dataset generator: determinism and label quality."""

import csv
import importlib.util
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_generator():
    """Load data/generate_dataset.py fresh by path (re-runs random.seed(42))."""
    path = os.path.join(PROJECT_ROOT, "data", "generate_dataset.py")
    spec = importlib.util.spec_from_file_location("generate_dataset_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_generation_is_deterministic(tmp_path):
    gen1 = _load_generator()
    gen1.generate_dataset(num_tickets=300, output_dir=str(tmp_path / "a"))
    gen2 = _load_generator()
    gen2.generate_dataset(num_tickets=300, output_dir=str(tmp_path / "b"))
    a = (tmp_path / "a" / "tickets.csv").read_bytes()
    b = (tmp_path / "b" / "tickets.csv").read_bytes()
    assert a == b


def test_schema_and_sizes(tmp_path):
    gen = _load_generator()
    gen.generate_dataset(num_tickets=2000, output_dir=str(tmp_path))
    rows = _read_rows(tmp_path / "tickets.csv")
    assert len(rows) == 2000
    assert set(rows[0].keys()) == {"ticket_id", "text", "category", "priority", "created_date"}
    assert all(r["priority"] in gen.PRIORITIES for r in rows)
    assert all(r["category"] in gen.CATEGORIES for r in rows)


def test_priority_is_text_correlated(tmp_path):
    """Priorities must be derived from content, not random noise."""
    gen = _load_generator()
    gen.generate_dataset(num_tickets=2000, output_dir=str(tmp_path))
    rows = _read_rows(tmp_path / "tickets.csv")

    hacked = [r["priority"] for r in rows if "hacked" in r["text"]]
    assert hacked, "no hacked-account tickets generated -- template changed?"
    assert set(hacked) <= {"Urgent", "High"}

    inquiry = {r["priority"] for r in rows if r["category"] == "General Inquiry"}
    assert inquiry <= {"Low", "Medium"}

    feature = {r["priority"] for r in rows if r["category"] == "Feature Request"}
    assert "Urgent" not in feature


def test_every_category_and_priority_present(tmp_path):
    gen = _load_generator()
    gen.generate_dataset(num_tickets=2000, output_dir=str(tmp_path))
    rows = _read_rows(tmp_path / "tickets.csv")
    assert {r["category"] for r in rows} == set(gen.CATEGORIES)
    assert {r["priority"] for r in rows} == set(gen.PRIORITIES)
