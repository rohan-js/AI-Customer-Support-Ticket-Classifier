"""
Priority Prediction Model: TF-IDF + Random Forest
===================================================
Predicts ticket priority level (Low / Medium / High / Urgent)
using Random Forest classifier with TF-IDF features.

Why Random Forest for priority?
- Handles the 4-class imbalanced problem well with class_weight='balanced'
- More robust to noise than single Logistic Regression
- Feature importance reveals which words signal urgency
- Ensemble method reduces overfitting on small data

Outputs:
- Saved model: models/saved/priority_rf_model.joblib
- Saved vectorizer: models/saved/priority_tfidf_vectorizer.joblib
- Saved label encoder: models/saved/priority_label_encoder.joblib
- Classification report + confusion matrix
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_dataset, prepare_data


def train_priority_model():
    """Train and evaluate Random Forest priority classifier."""

    # -- 1. Load and prepare data --
    df = load_dataset()
    data = prepare_data(df, target_col="priority")  # target = priority (not category)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_names = data["label_names"]

    # -- 2. TF-IDF Vectorization --
    print("\n[...] Fitting TF-IDF vectorizer for priority model...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # -- 3. Train Random Forest --
    # n_estimators=200 gives a good accuracy/speed tradeoff
    # class_weight='balanced' compensates for priority imbalance
    print("[...] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_tfidf, y_train)

    # -- 4. Evaluate --
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("PRIORITY MODEL RESULTS (TF-IDF + Random Forest)")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # -- 5. Feature Importance --
    feature_names = np.array(tfidf.get_feature_names_out())
    importances = model.feature_importances_
    top_n = 10
    top_indices = importances.argsort()[-top_n:][::-1]
    print(f"\nTop {top_n} Most Important Features for Priority:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:25s} importance: {importances[idx]:.4f}")

    # -- 6. Confusion Matrix --
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Priority Model - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "priority_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Confusion matrix saved -> {plot_path}")
    plt.close()

    # -- 7. Save model artifacts --
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "priority_rf_model.joblib"))
    joblib.dump(tfidf, os.path.join(save_dir, "priority_tfidf_vectorizer.joblib"))
    joblib.dump(data["label_encoder"], os.path.join(save_dir, "priority_label_encoder.joblib"))

    print(f"\n[OK] Priority model saved -> {save_dir}/")
    print(f"    - priority_rf_model.joblib")
    print(f"    - priority_tfidf_vectorizer.joblib")
    print(f"    - priority_label_encoder.joblib")

    return accuracy


if __name__ == "__main__":
    train_priority_model()
