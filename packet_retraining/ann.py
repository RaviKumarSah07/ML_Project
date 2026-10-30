"""
ann.py
------
A feed-forward Artificial Neural Network (MLP) implemented from scratch
using only NumPy. No scikit-learn / TensorFlow / PyTorch.

Architecture (default): Input(79) -> Dense(64, ReLU) -> Dense(32, ReLU)
                                    -> Dense(1, Sigmoid)
Loss: Binary Cross-Entropy
Optimizer: Mini-batch Gradient Descent (plain, fixed learning rate)

Every mathematical step is implemented explicitly and commented so it can
be explained in a viva: forward propagation, loss, backpropagation
(chain rule), and the gradient descent weight update.
"""

import pickle
import numpy as np


def _sigmoid(z):
    # Numerically stable sigmoid.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _relu(z):
    return np.maximum(0, z)


def _relu_derivative(z):
    return (z > 0).astype(z.dtype)


def _binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class ANN:
    """
    Configurable-depth MLP for binary classification.

    layer_sizes: list, e.g. [79, 64, 32, 1] -> input=79, two hidden layers
                 (64, 32) with ReLU, output=1 with Sigmoid.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1  # number of weight matrices

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            is_output_layer = (i == self.num_layers - 1)
            if is_output_layer:
                # Xavier/Glorot init for the sigmoid output layer.
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            else:
                # He init for ReLU hidden layers.
                std = np.sqrt(2.0 / fan_in)
                W = rng.normal(0, std, size=(fan_in, fan_out))
            self.weights.append(W)
            self.biases.append(np.zeros((1, fan_out)))

        self.train_loss_history = []
        self.val_loss_history = []

    # -- forward propagation --------------------------------------------------

    def forward(self, X):
        """
        Runs X through every layer, caching pre-activations (z) and
        activations (a) for use in backpropagation.
        Returns final sigmoid output (predicted probability of "Attack").
        """
        activations = [X]
        z_values = []
        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            z_values.append(z)
            is_output_layer = (i == self.num_layers - 1)
            a = _sigmoid(z) if is_output_layer else _relu(z)
            activations.append(a)
        self._cache = (activations, z_values)
        return a

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int32).ravel()

    # -- backpropagation + gradient descent ------------------------------------

    def backward(self, y_true):
        """
        Computes gradients of BCE loss w.r.t every weight/bias via the
        chain rule, then updates weights with plain gradient descent:
            W := W - learning_rate * dL/dW
        """
        activations, z_values = self._cache
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)

        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # dL/dz for the output layer: for BCE + sigmoid this simplifies to
        # (y_pred - y_true) -- a standard, well-known simplification of the
        # chain rule (dL/da * da/dz) for this specific loss/activation pair.
        y_pred = activations[-1]
        delta = (y_pred - y_true) / m  # shape (m, 1)

        for i in reversed(range(self.num_layers)):
            a_prev = activations[i]  # activation feeding INTO this layer
            grads_w[i] = a_prev.T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagate error to the previous layer through W^T,
                # then multiply by the derivative of THAT layer's activation
                # (ReLU derivative, since only hidden layers use ReLU).
                da_prev = delta @ self.weights[i].T
                delta = da_prev * _relu_derivative(z_values[i - 1])

        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # -- training loop -----------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256,
            early_stopping_patience=5, verbose=True):
        n = X_train.shape[0]
        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        patience_counter = 0
        rng = np.random.default_rng(0)

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = X_shuf[start:end], y_shuf[start:end]
                self.forward(Xb)
                self.backward(yb)

            train_pred = self.predict_proba(X_train)
            val_pred = self.predict_proba(X_val)
            train_loss = _binary_cross_entropy(y_train.reshape(-1, 1), train_pred)
            val_loss = _binary_cross_entropy(y_val.reshape(-1, 1), val_pred)
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if verbose:
                print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.5f})")
                    break

        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        return self

    # -- persistence -----------------------------------------------------

    def save(self, path: str):
        state = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "weights": self.weights,
            "biases": self.biases,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(state["layer_sizes"], learning_rate=state["learning_rate"])
        model.weights = state["weights"]
        model.biases = state["biases"]
        model.train_loss_history = state["train_loss_history"]
        model.val_loss_history = state["val_loss_history"]
        return model


if __name__ == "__main__":
    # Tiny smoke test on synthetic data to check shapes / that loss decreases.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    model = ANN([10, 8, 4, 1], learning_rate=0.1)
    model.fit(X, y, X, y, epochs=20, batch_size=32, verbose=True)
    preds = model.predict(X)
    acc = (preds == y).mean()
    print("Smoke-test train accuracy:", acc)
