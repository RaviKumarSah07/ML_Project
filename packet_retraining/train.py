import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ann import ANN
from evaluate import compute_metrics, plot_comparison, print_comparison_table, save_results_json
from packet_preprocessing import PACKET_FIELDS, PacketPreprocessor
from random_forest import RandomForest


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "ml_project" / "data"
ARTIFACTS_DIR = BASE_DIR / "saved_artifacts"
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.10
ANN_CONFIG = {
    "hidden_layers": [64, 32],
    "learning_rate": 0.01,
    "epochs": 50,
    "batch_size": 256,
    "early_stopping_patience": 5,
}
RF_CONFIG = {
    "n_trees": 50,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "bootstrap_sample_size": 25000,
}


def stratified_split(labels, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        validation_count = int(len(indices) * validation_fraction)
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])
    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def score(model, features, labels, train_time, config):
    start = time.time()
    predictions = model.predict(features)
    inference_time = time.time() - start
    metrics = compute_metrics(labels, predictions)
    metrics["train_time_sec"] = train_time
    metrics["inference_time_sec"] = inference_time
    metrics["config"] = config
    return metrics


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    train_data = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")
    test_data = pd.read_csv(DATA_DIR / "UNSW_NB15_testing-set.csv")
    train_indices, validation_indices = stratified_split(
        train_data["label"].to_numpy(), VALIDATION_FRACTION, RANDOM_SEED
    )

    preprocessor = PacketPreprocessor()
    x_train, y_train = preprocessor.fit_transform(train_data.iloc[train_indices])
    x_validation, y_validation = preprocessor.transform(train_data.iloc[validation_indices])
    x_test, y_test = preprocessor.transform(test_data)
    preprocessor.save(ARTIFACTS_DIR / "preprocessor.pkl")

    schema = {
        "input_fields": PACKET_FIELDS,
        "encoded_features": preprocessor.feature_names_,
        "selection_metric": "validation_f1_score",
        "decision_threshold": 0.5,
        "validation_fraction": VALIDATION_FRACTION,
        "random_seed": RANDOM_SEED,
    }
    with open(ARTIFACTS_DIR / "feature_schema.json", "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)

    print(f"Training rows: {len(x_train)}")
    print(f"Validation rows: {len(x_validation)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Input fields: {', '.join(PACKET_FIELDS)}")
    print(f"Encoded features: {x_train.shape[1]}")

    layer_sizes = [x_train.shape[1], *ANN_CONFIG["hidden_layers"], 1]
    ann = ANN(layer_sizes, learning_rate=ANN_CONFIG["learning_rate"], seed=RANDOM_SEED)
    start = time.time()
    ann.fit(
        x_train, y_train, x_validation, y_validation,
        epochs=ANN_CONFIG["epochs"],
        batch_size=ANN_CONFIG["batch_size"],
        early_stopping_patience=ANN_CONFIG["early_stopping_patience"],
        verbose=True,
    )
    ann_train_time = time.time() - start
    ann_config = {
        **ANN_CONFIG,
        "layer_sizes": layer_sizes,
        "epochs_actually_run": len(ann.train_loss_history),
    }
    ann.save(ARTIFACTS_DIR / "ann_weights.pkl")

    forest = RandomForest(**RF_CONFIG, seed=RANDOM_SEED)
    start = time.time()
    forest.fit(x_train, y_train, verbose=True)
    forest_train_time = time.time() - start
    forest.save(ARTIFACTS_DIR / "random_forest.pkl")

    validation_results = {
        "ANN": score(ann, x_validation, y_validation, ann_train_time, ann_config),
        "Random Forest": score(
            forest, x_validation, y_validation, forest_train_time, RF_CONFIG
        ),
    }
    print("\nValidation comparison:")
    print_comparison_table(validation_results)
    save_results_json(validation_results, ARTIFACTS_DIR / "validation_results.json")

    selected_model = max(
        validation_results,
        key=lambda name: validation_results[name]["f1_score"],
    )
    (ARTIFACTS_DIR / "selected_model.txt").write_text(selected_model, encoding="utf-8")
    print(f"Selected model by validation F1: {selected_model}")

    test_results = {
        "ANN": score(ann, x_test, y_test, ann_train_time, ann_config),
        "Random Forest": score(forest, x_test, y_test, forest_train_time, RF_CONFIG),
    }
    print("\nHeld-out test comparison:")
    print_comparison_table(test_results)
    save_results_json(test_results, ARTIFACTS_DIR / "evaluation_results.json")
    plot_comparison(test_results, ARTIFACTS_DIR / "model_comparison.png")
    print(f"Artifacts saved in: {ARTIFACTS_DIR}")
    return test_results


if __name__ == "__main__":
    main()
