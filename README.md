# AI Customer Support Ticket Classification & Auto-Reply System

An end-to-end ML system that automatically classifies customer support tickets, predicts priority levels, and generates intelligent auto-replies -- all running locally on CPU without any paid APIs. Built as a portfolio/educational project demonstrating the full train -> serve -> log -> BI pipeline.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

---

## Architecture

```
+-------------------------------------------------------------+
|                    FastAPI REST Service                       |
|                                                              |
|  POST /predict   ->  Category + Priority + Auto-Reply        |
|  GET  /health    ->  System health check                     |
|  GET  /analytics ->  Prediction analytics                    |
|  GET  /history   ->  Recent predictions                      |
+--------------------------------------------------------------+
|                     ML Pipeline                               |
|                                                              |
|  +------------------+  +------------------+                  |
|  | Category Model   |  | Priority Model   |                  |
|  | TF-IDF + LogReg  |  | TF-IDF + RF      |                  |
|  +------------------+  +------------------+                  |
|                                                              |
|  +------------------+  +------------------+                  |
|  | Transformer      |  | Reply Generator  |                  |
|  | DistilBERT       |  | Template-Based   |                  |
|  +------------------+  +------------------+                  |
+--------------------------------------------------------------+
|  SQLite DB  |  Analytics Engine  |  Synthetic Dataset         |
+--------------------------------------------------------------+
```

---

## Analytics & Visualizations

### Tickets per Category
![Tickets per Category](images/tickets_per_category.png)

### Priority Distribution
![Priority Distribution](images/priority_distribution.png)

### Daily Ticket Volume
![Daily Ticket Volume](images/daily_ticket_volume.png)

### Priority by Category Heatmap
![Priority by Category Heatmap](images/priority_by_category_heatmap.png)

### Baseline Model - Confusion Matrix
![Confusion Matrix](images/baseline_confusion_matrix.png)

---

## Project Structure

```
├── data/
│   ├── generate_dataset.py      # Synthetic data generator (5K tickets)
│   ├── tickets.csv              # Generated dataset (gitignored)
│   └── analytics_export.csv     # Power BI export (gitignored)
├── models/
│   ├── train_baseline.py        # TF-IDF + Logistic Regression
│   ├── train_transformer.py     # DistilBERT fine-tuning (optional)
│   ├── train_priority.py        # Random Forest priority model
│   └── saved/                   # Trained model artifacts (gitignored)
├── api/
│   └── main.py                  # FastAPI REST service
├── utils/
│   ├── preprocessing.py         # Text cleaning pipeline
│   ├── data_loader.py           # Dataset loading & splitting
│   ├── reply_generator.py       # Template-based auto-reply
│   ├── analytics.py             # Analytics & visualization
│   └── database.py              # SQLite prediction logging
├── tests/                       # Pytest suite
├── conftest.py                  # Makes project root importable in tests
├── images/                      # Charts for README
├── notebooks/                   # Jupyter exploration (optional)
├── requirements.txt             # Core pipeline dependencies
├── requirements-transformer.txt # Optional DistilBERT training extras
└── README.md
```

---

## Quick Start

### 1. Create an Environment (Python 3.13)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional, only for the DistilBERT training leg (large torch download):

```bash
pip install -r requirements-transformer.txt
```

### 3. Generate Dataset

```bash
python data/generate_dataset.py
```

This creates `data/tickets.csv` with 5,000 synthetic support tickets across 6 categories. Priorities are **derived from ticket content** (a security breach is Urgent, a feature request is Low), with 15% of labels shifted one level to simulate human labeling variance.

### 4. Train Models

```bash
# Baseline category classifier (fast -- seconds)
python models/train_baseline.py

# Priority prediction model (fast -- under a minute)
python models/train_priority.py

# Transformer model (optional -- slow on CPU)
python models/train_transformer.py --smoke   # ~5 min verification run
python models/train_transformer.py           # full fine-tune
```

Note: the first run of the preprocessing pipeline downloads NLTK
corpora (`punkt`, `punkt_tab`, `stopwords`, `wordnet`); afterwards they
are cached locally and no network is needed.

### 5. Generate Analytics

```bash
python utils/analytics.py
```

Produces charts and exports `data/analytics_export.csv` for Power BI.

### 6. Start API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 7. Test API

```bash
# Health check
curl http://localhost:8000/health

# Classify a ticket
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged $49.99 twice for my subscription"}'
```

Or visit **http://localhost:8000/docs** for the interactive Swagger UI.

### Run Tests

```bash
pytest
```

The suite covers preprocessing, the dataset generator (determinism +
text-priority correlation), the reply templates, the SQLite logging
layer, and the FastAPI endpoints (via `TestClient`, including the
graceful-degradation path when model files are missing).

---

## Models

| Model | Task | Algorithm | Test Accuracy | Notes |
|-------|------|-----------|---------------|-------|
| **Baseline** | Category Classification | TF-IDF + Logistic Regression | **100%** | Synthetic templates are keyword-separable |
| **Priority** | Priority Prediction | TF-IDF + Random Forest | **87%** | Learns real signals ("hacked account" -> Urgent); ~85% label-noise ceiling |
| **Transformer** | Category Classification | DistilBERT Fine-tuned | **100%** | ~38 min on CPU (3 epochs); matches the baseline exactly -- the synthetic task is too easy for a gap to show |
| **Reply** | Auto-Reply Generation | Template + Rule-Based | deterministic | Keyword-aware, priority-aware |

Honest caveats: because the dataset is synthetic, category separation is
trivial (100%) and the transformer comparison is mostly a demonstration
of the training loop -- real ticket data would show a meaningful gap.
The priority task is learnable but capped near the injected label-noise
rate, which is intentional: it produces realistic confusion (Urgent
recall 0.52) instead of fake perfection.

### Categories
- Billing Issue
- Technical Problem
- Account Access
- Refund Request
- Feature Request
- General Inquiry

### Priority Levels
- Low -> Medium -> High -> Urgent

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Classify ticket + predict priority + generate reply |
| `GET` | `/health` | System health and model status |
| `GET` | `/analytics` | Aggregate prediction analytics (from logged predictions) |
| `GET` | `/history` | Recent prediction history |

### Example Response

```json
{
  "category": "Billing Issue",
  "priority": "High",
  "reply": "This has been marked as a high-priority issue.  \n\nHello,  \n\nWe appreciate you bringing this billing issue to our attention regarding the charge of $49.99. ...",
  "timestamp": "2026-09-16T10:30:00"
}
```

If model artifacts are missing, the API degrades gracefully:
`/predict` returns `"category": "Unknown (model not loaded)"` and
`"priority": "Medium"` instead of crashing.

---

## Power BI Integration

The `data/analytics_export.csv` includes:
- All ticket fields
- Numeric priority (1-4) for calculations
- Day of week and month for time analysis
- Ready to import directly into Power BI / Tableau / Excel

---

## Tech Stack

- **Python 3.13**
- **scikit-learn** -- TF-IDF, Logistic Regression, Random Forest
- **HuggingFace Transformers** -- DistilBERT fine-tuning (optional)
- **FastAPI** -- REST API framework
- **SQLite** -- Prediction logging database
- **Pandas / NumPy** -- Data processing
- **Matplotlib / Seaborn** -- Visualization
- **NLTK** -- Text preprocessing
- **Pytest** -- Test suite

---

## License

This project is for educational and portfolio purposes.
