"""
Baseline Category Classifier: TF-IDF + Logistic Regression
============================================================
Fast, interpretable baseline model for ticket category classification.

Why TF-IDF + LogReg?
- Fast to train (seconds, not hours)
- Strong baseline for text classification
- Interpretable features (top TF-IDF terms per class)
- Serves as comparison benchmark for transformer model
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_dataset, prepare_data


def train_baseline():
    """Train and evaluate TF-IDF + Logistic Regression baseline model."""

    df = load_dataset()
    data = prepare_data(df, target_col="category")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_names = data["label_names"]

    print("\n[...] Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"    Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"    Feature matrix: {X_train_tfidf.shape}")

    print("\n[...] Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("BASELINE MODEL RESULTS (TF-IDF + Logistic Regression)")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Baseline Model - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "baseline_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Confusion matrix saved -> {plot_path}")
    plt.close()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "category_baseline_model.joblib"))
    joblib.dump(tfidf, os.path.join(save_dir, "category_tfidf_vectorizer.joblib"))
    joblib.dump(data["label_encoder"], os.path.join(save_dir, "category_label_encoder.joblib"))

    print(f"[OK] Model artifacts saved -> {save_dir}/")
    print(f"    - category_baseline_model.joblib")
    print(f"    - category_tfidf_vectorizer.joblib")
    print(f"    - category_label_encoder.joblib")

    return accuracy


if __name__ == "__main__":
    train_baseline()
