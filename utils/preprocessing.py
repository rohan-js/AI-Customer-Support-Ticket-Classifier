"""
Text Preprocessing Pipeline
============================
Provides text cleaning utilities for NLP tasks:
- Lowercase normalization
- Special character / punctuation removal
- Stopword removal (NLTK)
- Lemmatization (WordNet)

Design choice: Using NLTK over spaCy for lighter memory footprint.
All functions are pure (no side effects) and composable.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Required NLTK data (one-time download, cached after first run)
_REQUIRED_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet"]


def _ensure_nltk_data():
    """Download NLTK data, failing with an actionable message if unavailable."""
    for package in _REQUIRED_NLTK_PACKAGES:
        try:
            nltk.download(package, quiet=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download NLTK data '{package}' ({exc}). "
                "An internet connection is required on the first run; the data is "
                "cached afterwards and no connection is needed again."
            ) from exc


_ensure_nltk_data()

# Initialize once at module level to avoid repeated object creation
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single text string.

    Steps:
        1. Lowercase
        2. Remove special characters (keep letters, numbers, spaces)
        3. Tokenize
        4. Remove stopwords
        5. Lemmatize each token
        6. Rejoin into cleaned string

    Args:
        text: Raw input text

    Returns:
        Cleaned, preprocessed text string
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove special characters and extra whitespace
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 3: Tokenize
    tokens = word_tokenize(text)

    # Step 4 & 5: Remove stopwords and lemmatize in one pass (efficient)
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]

    # Step 6: Rejoin
    return " ".join(tokens)


def batch_clean(texts: list) -> list:
    """Apply clean_text to a list of strings. Useful for DataFrame columns."""
    return [clean_text(t) for t in texts]


if __name__ == "__main__":
    sample = "I was CHARGED $49.99 twice!!! Please fix this ASAP. My account is user123@gmail.com"
    print(f"Original: {sample}")
    print(f"Cleaned:  {clean_text(sample)}")
