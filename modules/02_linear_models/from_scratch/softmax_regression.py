"""Softmax regression from scratch."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along class axis."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1.0
    log_probs = np.log(probs + 1e-12)
    return float(-np.mean(np.sum(one_hot * log_probs, axis=1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y))


def train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float = 0.5,
    n_epochs: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Train softmax regression with full-batch GD."""
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    W = rng.normal(scale=0.01, size=(n_features, n_classes))
    b = np.zeros(n_classes)
    n_samples = X.shape[0]
    losses: list[float] = []
    accs: list[float] = []

    for _ in range(n_epochs):
        logits = X @ W + b
        probs = softmax(logits)
        loss = cross_entropy(probs, y, n_classes)
        losses.append(loss)
        accs.append(accuracy(probs, y))

        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1.0
        grad_logits = (probs - one_hot) / n_samples
        grad_W = X.T @ grad_logits
        grad_b = np.sum(grad_logits, axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

    return W, b, losses, accs


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from lib.synthetic_data import multiclass_blobs

    X, y = multiclass_blobs(n_classes=3)
    W, b, losses, accs = train_softmax(X, y, n_classes=3)
    print(f"Final loss: {losses[-1]:.4f}, accuracy: {accs[-1]:.4f}")
