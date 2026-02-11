"""
Transformer Category Classifier: DistilBERT Fine-Tuning
========================================================
Fine-tunes DistilBERT for ticket category classification using HuggingFace Trainer.

Why DistilBERT over full BERT?
- 40% fewer parameters (66M vs 110M)
- 60% faster inference
- ~97% of BERT's performance on most NLU tasks
- Practical for CPU training (this project runs without GPU)

Why transformers work better than TF-IDF:
- Contextual embeddings understand word meaning based on surrounding words
  ("bank" in "river bank" vs "bank account")
- Pre-trained on massive corpora -- transfers general language understanding
- Handles unseen vocabulary through subword tokenization (WordPiece)
- Captures long-range dependencies in text that n-grams miss

Training is configured for CPU:
- Small batch size (8)
- 3 epochs (sufficient for fine-tuning a pre-trained model)
- FP32 (no mixed precision -- CPU doesn't support FP16 well)

Outputs:
- Saved model: models/saved/category_transformer/
- Classification report + confusion matrix comparison
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_dataset, prepare_data


# -----------------------------------------------
# Custom Dataset class for HuggingFace Trainer
# -----------------------------------------------
class TicketDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized ticket data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    """Custom metrics function for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def train_transformer():
    """Fine-tune DistilBERT for ticket category classification."""

    # -- 1. Load and prepare data --
    df = load_dataset()
    data = prepare_data(df, target_col="category")

    # Transformer uses RAW text (its own tokenizer handles preprocessing)
    X_train_raw = data["X_train_raw"]
    X_test_raw = data["X_test_raw"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_names = data["label_names"]
    num_labels = len(label_names)

    # -- 2. Tokenize with DistilBERT tokenizer --
    print("\n[...] Tokenizing with DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # max_length=128 is sufficient for short support tickets
    train_encodings = tokenizer(X_train_raw, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test_raw, truncation=True, padding=True, max_length=128)

    train_dataset = TicketDataset(train_encodings, y_train.tolist())
    test_dataset = TicketDataset(test_encodings, y_test.tolist())

    # -- 3. Load pre-trained DistilBERT --
    print("[...] Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )

    # -- 4. Configure training (CPU-optimized) --
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved", "category_transformer")
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,            # enough for fine-tuning pre-trained model
        per_device_train_batch_size=8, # small batch for CPU memory
        per_device_eval_batch_size=16,
        warmup_steps=100,              # gradual learning rate warmup
        weight_decay=0.01,             # L2 regularization
        logging_steps=50,
        eval_strategy="epoch",         # evaluate after each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,   # keep best checkpoint
        metric_for_best_model="accuracy",
        report_to="none",             # disable wandb/tensorboard
        no_cuda=True,                  # force CPU training
    )

    # -- 5. Train --
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n[...] Fine-tuning DistilBERT (this may take a while on CPU)...")
    trainer.train()

    # -- 6. Evaluate --
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("TRANSFORMER MODEL RESULTS (DistilBERT Fine-Tuned)")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # -- 7. Confusion Matrix --
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title("Transformer Model - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    plot_path = os.path.join(plots_dir, "transformer_confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Confusion matrix saved -> {plot_path}")
    plt.close()

    # -- 8. Save model + tokenizer --
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save label mapping for inference
    label_map = {i: label for i, label in enumerate(label_names)}
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[OK] Transformer model saved -> {save_dir}/")

    # -- 9. Print comparison summary --
    print("\n" + "=" * 60)
    print("WHY TRANSFORMER WORKS BETTER")
    print("=" * 60)
    print("""
    1. CONTEXTUAL UNDERSTANDING: DistilBERT generates context-aware
       embeddings. 'charged twice' and 'double charge' are understood
       as semantically similar, unlike TF-IDF which treats them as
       different tokens.

    2. TRANSFER LEARNING: Pre-trained on BookCorpus + Wikipedia (3.3B
       words), the model already understands English grammar, sentiment,
       and common patterns before seeing any support tickets.

    3. SUBWORD TOKENIZATION: WordPiece handles unseen words by breaking
       them into known subwords. 'unsubscribed' -> 'un' + '##subscribed',
       preserving meaning even for novel vocabulary.

    4. ATTENTION MECHANISM: Self-attention captures relationships between
       any two words regardless of distance. In 'refund because defective',
       the model links 'refund' with 'defective' across the sentence.
    """)

    return accuracy


if __name__ == "__main__":
    train_transformer()
