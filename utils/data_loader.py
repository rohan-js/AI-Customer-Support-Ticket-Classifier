"""
Data Loader & Splitter
=======================
Handles loading the ticket dataset and splitting into train/test sets.

Key features:
- Stratified split to maintain category/priority distribution in both sets
- Label encoding for ML models
- Returns both raw and preprocessed text for different model needs
  (TF-IDF needs preprocessed, transformers need raw)
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import batch_clean


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Load ticket dataset from CSV.

    Args:
        data_path: Path to tickets.csv. Defaults to data/tickets.csv relative to project root.

    Returns:
        DataFrame with all ticket columns
    """
    if data_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, "data", "tickets.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run 'python data/generate_dataset.py' first."
        )

    df = pd.read_csv(data_path)
    print(f"[OK] Loaded {len(df)} tickets from {data_path}")
    print(f"    Categories: {df['category'].nunique()} | Priorities: {df['priority'].nunique()}")
    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "category",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Preprocess text, encode labels, and split into train/test sets.

    Args:
        df: Raw DataFrame with 'text' and target column
        target_col: Column to predict ('category' or 'priority')
        test_size: Fraction of data for test set
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - X_train, X_test: preprocessed text lists
            - X_train_raw, X_test_raw: raw text lists (for transformers)
            - y_train, y_test: encoded label arrays
            - label_encoder: fitted LabelEncoder instance
            - label_names: list of original label names
    """
    # Apply text preprocessing for traditional ML models
    print("[...] Preprocessing text (lemmatization, stopwords removal)...")
    df = df.copy()
    df["clean_text"] = batch_clean(df["text"].tolist())

    # Encode target labels as integers
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[target_col])

    # Stratified train/test split -- maintains class distribution
    X_train_raw, X_test_raw, X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),       # raw text for transformers
        df["clean_text"].tolist(), # preprocessed text for TF-IDF
        df["label"].values,        # encoded labels
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"].values,  # ensure proportional split
    )

    print(f"[OK] Split: {len(X_train)} train / {len(X_test)} test (stratified)")
    print(f"    Labels: {list(label_encoder.classes_)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "label_names": list(label_encoder.classes_),
    }


if __name__ == "__main__":
    df = load_dataset()
    data = prepare_data(df, target_col="category")
    print(f"\nSample cleaned text: {data['X_train'][0]}")
    print(f"Sample label: {data['y_train'][0]} -> {data['label_names'][data['y_train'][0]]}")
