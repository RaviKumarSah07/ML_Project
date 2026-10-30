# Network Intrusion Detection — ML Phase (UNSW-NB15)

Implements and compares two from-scratch ML algorithms on UNSW-NB15 for binary
intrusion detection (Normal vs Attack), per the academic requirement that
no ready-made classifiers (`MLPClassifier`, `RandomForestClassifier`,
`DecisionTreeClassifier`, TensorFlow/PyTorch) be used.

## Files

| File | Purpose |
|---|---|
| `preprocessing.py` | Cleans data, one-hot encodes categoricals, scales numerics. Fit on train only. |
| `ann.py` | Feed-forward ANN/MLP — forward prop, backprop, mini-batch gradient descent (pure NumPy). |
| `random_forest.py` | Decision Tree (Gini, recursive splitting) + Random Forest (bootstrap + majority vote), pure NumPy. |
| `evaluate.py` | Accuracy/precision/recall/F1/confusion matrix, comparison table & chart — all from scratch. |
| `train.py` | Orchestrates the full pipeline end to end. |
| `main.py` | Entry point — just calls `train.py`. |
| `data/` | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` (place here to reproduce). |
| `saved_artifacts/` | Output: trained weights, RF model, preprocessor config, results, chart. |

## Run

```
pip install numpy pandas matplotlib
python3 main.py
```

## Key design decisions (for the viva)

- **Dropped columns:** `id` (row index, not a feature) and `attack_cat`
  (multi-class label that perfectly determines `label` — including it
  would be data leakage).
- **Categorical encoding:** `proto` has 133 raw categories in the
  training data; the top 15 (93.4% of rows) are kept individually and
  everything else — including any protocol never seen during training —
  is bucketed into `Other`. `service` and `state` keep their full
  vocabularies plus an `Other` bucket for the same reason (the test set
  contains `state` values, `CLO`/`ACC`, that never appear in training —
  this is exactly what `Other` catches safely instead of crashing or
  silently mis-encoding).
- **Numeric scaling:** 31 of 39 numeric columns had skewness > 1.0
  (byte counts, rates, durations, etc.), so those get `log1p` before
  z-score standardization; all scaling parameters are fit on the
  training data only and reused on test data.
- **ANN:** 79 → 64 (ReLU) → 32 (ReLU) → 1 (Sigmoid), binary
  cross-entropy loss, mini-batch gradient descent (batch=256, lr=0.01),
  up to 50 epochs with early stopping (patience=5) on a held-out
  validation split.
- **Random Forest:** 50 trees, Gini-impurity splits, max_depth=12,
  min_samples_leaf=5, √79≈9 random features considered per split,
  bootstrap samples capped at 25,000 rows/tree. This cap (rather than a
  full 175K-row bootstrap per tree) was a deliberate, benchmarked
  tradeoff: a single tree at this size took 0.45s to build on this
  data, keeping the full 50-tree forest under 30 seconds.
- **Validation split:** a stratified 90/10 split carved from the
  *training* file only (preserves the Normal/Attack ratio); the actual
  test file is never touched until final evaluation.
- **Why recall matters more than accuracy here:** training data is
  ~68% Attack / 32% Normal, so a model that always predicts "Attack"
  would already score ~68% accuracy without learning anything. That's
  why precision, recall (particularly Attack recall — missed attacks
  are the costly error), F1, and the confusion matrix are reported
  alongside accuracy, not instead of it.

## Actual results (this run, on the real held-out test set, 82,332 records)

| Metric | ANN | Random Forest |
|---|---|---|
| Accuracy | 0.8563 | 0.8529 |
| Precision | 0.7982 | 0.7936 |
| Recall (Attack) | 0.9891 | 0.9906 |
| F1-score | 0.8834 | 0.8812 |
| Specificity (Normal recall) | 0.6935 | 0.6843 |
| Training time (s) | 41.47 | 22.92 |
| Inference time (s) | 0.12 | 4.68 |

**Selected for deployment:** ANN (higher F1-score), though the two models
are close — Random Forest trains faster and has marginally higher attack
recall, while the ANN has faster inference and slightly better overall
balance. Both catch ~99% of actual attacks in this test split.

## Reproducing / loading saved artifacts

```python
from preprocessing import Preprocessor
from ann import ANN
from random_forest import RandomForest

pre = Preprocessor.load("saved_artifacts/preprocessor.pkl")
ann = ANN.load("saved_artifacts/ann_weights.pkl")
rf = RandomForest.load("saved_artifacts/random_forest.pkl")

X_new, y_new = pre.transform(new_dataframe)  # same encoding/scaling as training
predictions = ann.predict(X_new)
```

## Explicitly out of scope for this phase

Frontend, backend/API, packet capture, real-time feature extraction,
authentication, database, deployment infrastructure — per Section 18,
these come after this ML phase is validated.
