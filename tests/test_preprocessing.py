"""Tests for the text preprocessing pipeline."""

from utils.preprocessing import clean_text, batch_clean


def test_lowercases_and_strips_punctuation():
    cleaned = clean_text("Hello, WORLD!!!")
    assert cleaned == "hello world"


def test_keeps_digits_but_not_currency_symbols():
    cleaned = clean_text("I was charged $49.99 twice.")
    assert "4999" in cleaned
    assert "$" not in cleaned
    assert "." not in cleaned


def test_removes_stopwords_short_tokens():
    # "i", "was", "a" are stopwords / too-short tokens
    cleaned = clean_text("I was charged twice")
    assert "i" not in cleaned.split()
    assert "was" not in cleaned.split()


def test_lemmatizes_tokens():
    # WordNet lemmatization reduces "charges" -> "charge" in noun contexts
    assert "charge" in clean_text("billing charges")


def test_empty_and_single_word_inputs():
    assert clean_text("") == ""
    assert clean_text("Refund") == "refund"


def test_batch_clean_preserves_length_and_order():
    texts = ["I was charged twice", "Please refund my money"]
    out = batch_clean(texts)
    assert len(out) == 2
    assert out[0] == clean_text(texts[0])
    assert out[1] == clean_text(texts[1])
