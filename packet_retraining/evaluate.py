"""
evaluate.py
-----------
Computes accuracy, precision, recall, F1, and the confusion matrix for a
model's predictions, all from scratch (no sklearn.metrics), and produces
a side-by-side comparison table + chart for ANN vs Random Forest.

Because the dataset is imbalanced (68% Attack / 32% Normal in training),
accuracy alone is not trusted here -- recall on the Attack class is
reported explicitly, since missing a real attack is the costliest error
for an intrusion-detection system.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred):
    """
    Returns [[TN, FP], [FN, TP]] with class 0 = Normal, class 1 = Attack.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = Attack recall (sensitivity)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Normal recall

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_attack": float(recall),
        "f1_score": float(f1),
        "specificity_normal_recall": float(specificity),
        "confusion_matrix": cm.tolist(),
    }


def print_comparison_table(results: dict):
    """results = {'ANN': {...metrics + timing}, 'Random Forest': {...metrics + timing}}"""
    metrics = ["accuracy", "precision", "recall_attack", "f1_score", "specificity_normal_recall",
               "train_time_sec", "inference_time_sec"]
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall_attack": "Recall (Attack)",
        "f1_score": "F1-score",
        "specificity_normal_recall": "Specificity (Normal recall)",
        "train_time_sec": "Training time (s)",
        "inference_time_sec": "Inference time (s)",
    }
    model_names = list(results.keys())
    col_width = 18
    header = "Metric".ljust(28) + "".join(name.ljust(col_width) for name in model_names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = labels[m].ljust(28)
        for name in model_names:
            val = results[name].get(m, float("nan"))
            row += f"{val:.4f}".ljust(col_width) if "time" not in m else f"{val:.2f}".ljust(col_width)
        print(row)
    print()
    for name in model_names:
        cm = np.array(results[name]["confusion_matrix"])
        print(f"{name} confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)
        print()


def plot_comparison(results: dict, out_path: str):
    metrics = ["accuracy", "precision", "recall_attack", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall (Attack)", "F1-score"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ANN vs Random Forest — UNSW-NB15 Intrusion Detection")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_results_json(results: dict, out_path: str):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
