# ML Pipeline: Text Embeddings + Logistic Regression

A production-ready binary classification pipeline that combines **numerical features**, **categorical features**, and **free-text messages** into a single model. Text is embedded using a pretrained sentence transformer, and a regularised Logistic Regression classifier is trained and evaluated with full hyperparameter optimisation.

---

## Features

- **Pretrained text embeddings** via `sentence-transformers` (all-MiniLM-L6-v2) — no training from scratch
- **Embedding-aware feature selection** — text dimensions treated as an atomic group, not independent features
- **GridSearchCV optimisation** across L1, L2, and ElasticNet penalties with C < 1
- **WITH vs WITHOUT text comparison** — two fully independent pipelines, paired t-test for significance
- **Configurable dataset** — variable number of numerical features and controllable text noise level
- **Visualisation dashboards** — 6-panel results plot and 6-panel comparison plot

---

## Installation

```bash
pip install sentence-transformers scikit-learn numpy pandas matplotlib scipy
```

---

## Quick Start

```bash
python ml_pipeline.py
```

This runs the full pipeline with defaults: 1,000 samples, 3 numerical features, no text noise.

To customise, edit the single `generate_dataset` call in `main()`:

```python
df = generate_dataset(
    n_samples=1000,
    n_num_features=5,    # 3, 4, 5, 10, or any positive integer
    text_noise_pct=0.1,  # 0.0 = clean, 0.2 = 20% of messages corrupted
)
```

---

## Configuration

Column names are declared once at the top of the script:

```python
NUM_COLS   = [f"num_feature_{i+1}" for i in range(N_NUM_FEATURES)]
CAT_COLS   = ["cat_feature_A", "cat_feature_B"]
TEXT_COL   = "text_message"
TARGET_COL = "target"
```

To use your own dataset, update these names and replace the `generate_dataset()` call with your own DataFrame loading code.

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Dataset | Synthetic data generation with configurable numerical features and optional text noise |
| 2. Embedding | Pretrained sentence transformer (frozen) → 384-dim → PCA → 10-dim |
| 3. Preprocessing | StandardScaler (numerical) + OneHotEncoder (categorical) + text embeddings concatenated |
| 4. Feature Selection | Mutual information — text embedding treated as one group (keep-all or drop-all) |
| 5. Optimisation | GridSearchCV over L1 / L2 / ElasticNet, C ∈ {0.001, 0.01, 0.1}, 5-fold stratified CV |
| 6. Evaluation | Accuracy, Precision, Recall, F1, AUC on held-out test set |
| 7. Visualisation | 6-panel dashboard saved to `ml_pipeline_results.png` |
| 8. Comparison | Two independent pipelines (WITH vs WITHOUT text), paired t-test on CV AUC folds |

---

## Text Embedding Design

The embedder strictly separates fit from transform to prevent data leakage:

- `fit_transform(train_texts)` — encodes training sentences with the frozen pretrained model, then **fits PCA on training embeddings only**
- `transform(test_texts)` — applies the same frozen model and frozen PCA; test data has zero influence on any weights or projection axes

To swap the model, pass a different `model_name` to `TextEmbedder`:

| Model | Params | Output Dim | Notes |
|-------|--------|-----------|-------|
| `all-MiniLM-L6-v2` (default) | 22M | 384 | Best speed/quality balance |
| `all-mpnet-base-v2` | 110M | 768 | Highest quality, slower |
| `paraphrase-MiniLM-L3-v2` | 17M | 384 | Ultra-fast, good for short texts |
| `all-distilroberta-v1` | 82M | 768 | Strong on domain-specific language |

---

## Feature Selection

Standard `SelectKBest` treats each `text_emb_*` dimension independently — this destroys the geometric structure of the embedding space. This pipeline scores the embedding as a single group using mean mutual information across all dimensions, and either includes all 10 dimensions or excludes all 10 as one selection decision.

---

## Outputs

| File | Contents |
|------|----------|
| `ml_pipeline_results.png` | 6-panel dashboard for the WITH-text model |
| `comparison_results.png` | 6-panel WITH vs WITHOUT text comparison |

---

## Project Structure

```
ml_pipeline.py       # Main pipeline script
README.md            # This file
```

---

## Requirements

- Python 3.10+
- sentence-transformers
- scikit-learn
- numpy
- pandas
- matplotlib
- scipy
