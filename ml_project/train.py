import os
import time

import numpy as np

from preprocessing import Preprocessor, load_raw
from ann import ANN
from random_forest import RandomForest
from evaluate import (
    compute_metrics,
    print_comparison_table,
    plot_comparison,
    save_results_json,
)

DATA_DIR = "data"
ARTIFACTS_DIR = "saved_artifacts"


ANN_CONFIG = dict(
    hidden_layers=[64, 32],
    learning_rate=0.01,
    epochs=50,
    batch_size=256,
    early_stopping_patience=5,
)
RF_CONFIG = dict(
    n_trees=50,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    bootstrap_sample_size=25000,
)
VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42


def stratified_split(X, y, val_fraction, seed):
    rng = np.random.default_rng(seed)
    val_idx_parts = []
    train_idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(len(cls_idx) * val_fraction)
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])
    val_idx = np.concatenate(val_idx_parts)
    train_idx = np.concatenate(train_idx_parts)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Load raw data")
    print("=" * 70)
    train_df, test_df = load_raw(
        os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
        os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv"),
    )
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing (fit on train only)")
    print("=" * 70)
    pre = Preprocessor()
    X_train_full, y_train_full = pre.fit_transform(train_df)
    X_test, y_test = pre.transform(test_df)
    pre.save(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    print(
        f"Feature matrix: {X_train_full.shape[1]} columns "
        f"({len(pre.categorical_vocab)} categorical groups one-hot encoded, "
        f"{len(pre.numeric_columns)} numeric columns scaled)"
    )
    print(f"X_train_full: {X_train_full.shape}, X_test: {X_test.shape}")

    print("\n" + "=" * 70)
    print("STEP 3: Stratified train/validation split (from training data only)")
    print("=" * 70)
    train_idx, val_idx = stratified_split(
        X_train_full, y_train_full, VALIDATION_FRACTION, RANDOM_SEED
    )
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]
    print(f"X_train: {X_train.shape} (attack rate={y_train.mean():.3f})")
    print(f"X_val  : {X_val.shape} (attack rate={y_val.mean():.3f})")
    print(
        f"X_test : {X_test.shape} (attack rate={y_test.mean():.3f})  <- held out, untouched until final eval"
    )

    results = {}

    print("\n" + "=" * 70)
    print("STEP 4: Train ANN (from scratch)")
    print("=" * 70)
    layer_sizes = [X_train.shape[1]] + ANN_CONFIG["hidden_layers"] + [1]
    print(f"Architecture: {layer_sizes}")
    ann = ANN(layer_sizes, learning_rate=ANN_CONFIG["learning_rate"], seed=RANDOM_SEED)

    t0 = time.time()
    ann.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=ANN_CONFIG["epochs"],
        batch_size=ANN_CONFIG["batch_size"],
        early_stopping_patience=ANN_CONFIG["early_stopping_patience"],
        verbose=True,
    )
    ann_train_time = time.time() - t0

    t0 = time.time()
    ann_test_pred = ann.predict(X_test)
    ann_inference_time = time.time() - t0

    ann_metrics = compute_metrics(y_test, ann_test_pred)
    ann_metrics["train_time_sec"] = ann_train_time
    ann_metrics["inference_time_sec"] = ann_inference_time
    ann_metrics["config"] = {
        **ANN_CONFIG,
        "layer_sizes": layer_sizes,
        "epochs_actually_run": len(ann.train_loss_history),
    }
    results["ANN"] = ann_metrics
    ann.save(os.path.join(ARTIFACTS_DIR, "ann_weights.pkl"))
    print(
        f"ANN training time: {ann_train_time:.2f}s | inference time on test set: {ann_inference_time:.3f}s"
    )

    print("\n" + "=" * 70)
    print("STEP 5: Train Random Forest (from scratch)")
    print("=" * 70)
    rf = RandomForest(
        n_trees=RF_CONFIG["n_trees"],
        max_depth=RF_CONFIG["max_depth"],
        min_samples_split=RF_CONFIG["min_samples_split"],
        min_samples_leaf=RF_CONFIG["min_samples_leaf"],
        bootstrap_sample_size=RF_CONFIG["bootstrap_sample_size"],
        seed=RANDOM_SEED,
    )
    t0 = time.time()
    rf.fit(X_train, y_train, verbose=True)
    rf_train_time = time.time() - t0

    t0 = time.time()
    rf_test_pred = rf.predict(X_test)
    rf_inference_time = time.time() - t0

    rf_metrics = compute_metrics(y_test, rf_test_pred)
    rf_metrics["train_time_sec"] = rf_train_time
    rf_metrics["inference_time_sec"] = rf_inference_time
    rf_metrics["config"] = RF_CONFIG
    results["Random Forest"] = rf_metrics
    rf.save(os.path.join(ARTIFACTS_DIR, "random_forest.pkl"))
    print(
        f"RF training time: {rf_train_time:.2f}s | inference time on test set: {rf_inference_time:.3f}s"
    )

    print("\n" + "=" * 70)
    print("STEP 6: Evaluation & Comparison (on held-out test set)")
    print("=" * 70)
    print_comparison_table(results)

    save_results_json(results, os.path.join(ARTIFACTS_DIR, "evaluation_results.json"))
    plot_comparison(results, os.path.join(ARTIFACTS_DIR, "model_comparison.png"))

    best_model = max(results, key=lambda k: results[k]["f1_score"])
    print(f"Selected model for deployment (highest F1-score): {best_model}")
    with open(os.path.join(ARTIFACTS_DIR, "selected_model.txt"), "w") as f:
        f.write(best_model)

    print("\nAll artifacts saved to:", os.path.abspath(ARTIFACTS_DIR))
    return results


if __name__ == "__main__":
    main()
