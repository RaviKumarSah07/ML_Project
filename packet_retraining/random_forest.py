"""
random_forest.py
-----------------
Decision Tree and Random Forest implemented from scratch using only NumPy.
No sklearn.tree / sklearn.ensemble.

DecisionTree:
    - Recursive binary splitting.
    - Splitting criterion: Gini impurity.
    - At each node, only a random subset of features is considered for
      splitting (max_features) -- this randomness is what makes a FOREST
      of these trees decorrelated from each other.
    - Stopping conditions: max_depth, min_samples_split, min_samples_leaf,
      or a pure node (Gini == 0).

RandomForest:
    - Trains `n_trees` DecisionTrees, each on a bootstrap sample
      (sampling WITH replacement) of the training data.
    - Final prediction = majority vote across all trees.
    - Predicted "probability" = fraction of trees voting Attack.
"""

import pickle
import numpy as np


def _gini(y):
    if y.size == 0:
        return 0.0
    p = np.bincount(y.astype(np.int64), minlength=2) / y.size
    return 1.0 - np.sum(p ** 2)


class _Node:
    __slots__ = ("is_leaf", "prediction", "feature_idx", "threshold", "left", "right")

    def __init__(self):
        self.is_leaf = False
        self.prediction = None
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=10, min_samples_leaf=5,
                 max_features=None, n_split_candidates=16, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        # For continuous features, evaluating every unique value as a split
        # point is expensive. We sample a fixed number of candidate
        # thresholds (percentiles of the feature's values at this node)
        # instead -- a standard, well-documented speed/accuracy tradeoff
        # for from-scratch trees on continuous data.
        self.n_split_candidates = n_split_candidates
        self.rng = np.random.default_rng(seed)
        self.root = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(self.n_features_)))
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        node = _Node()
        n_samples = X.shape[0]
        gini = _gini(y)

        # Stopping conditions
        if (gini == 0.0 or depth >= self.max_depth
                or n_samples < self.min_samples_split):
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        feature_idx, threshold, split_gain = self._best_split(X, y)

        if feature_idx is None or split_gain <= 0:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = self._majority_class(y)
            return node

        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _majority_class(self, y):
        counts = np.bincount(y.astype(np.int64), minlength=2)
        return int(np.argmax(counts))

    def _best_split(self, X, y):
        n_samples = X.shape[0]
        parent_gini = _gini(y)

        candidate_features = self.rng.choice(
            self.n_features_, size=min(self.max_features, self.n_features_), replace=False
        )

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feat in candidate_features:
            col = X[:, feat]
            # Candidate thresholds = percentiles of this feature's values,
            # so we don't need to test every unique value (fast, still
            # covers the meaningful split points).
            percentiles = np.linspace(5, 95, self.n_split_candidates)
            thresholds = np.unique(np.percentile(col, percentiles))

            for t in thresholds:
                left_mask = col <= t
                n_left = left_mask.sum()
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = _gini(y[left_mask])
                gini_right = _gini(y[~left_mask])
                weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.prediction


class RandomForest:
    def __init__(self, n_trees=20, max_depth=12, min_samples_split=10,
                 min_samples_leaf=5, max_features=None, bootstrap_sample_size=25000,
                 seed=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_sample_size = bootstrap_sample_size
        self.seed = seed
        self.trees = []

    def fit(self, X, y, verbose=True):
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        sample_size = min(self.bootstrap_sample_size, n)
        self.trees = []

        for i in range(self.n_trees):
            idx = rng.integers(0, n, size=sample_size)  # bootstrap: sample WITH replacement
            X_boot, y_boot = X[idx], y[idx]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=self.seed + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            if verbose:
                print(f"  Tree {i + 1}/{self.n_trees} built.")
        return self

    def predict_proba(self, X):
        """Fraction of trees voting 'Attack' (class 1) for each sample."""
        votes = np.zeros(X.shape[0])
        for tree in self.trees:
            votes += tree.predict(X)
        return votes / len(self.trees)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Smoke test on synthetic data.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    rf = RandomForest(n_trees=5, max_depth=6, bootstrap_sample_size=1500)
    rf.fit(X, y)
    preds = rf.predict(X)
    acc = (preds == y).mean()
    print("Smoke-test train accuracy:", acc)
