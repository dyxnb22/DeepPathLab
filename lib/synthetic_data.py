"""Synthetic datasets for regression, classification, and MLP experiments.

All generators use numpy and accept a seed for reproducible experiments
without external dataset downloads.
"""

from __future__ import annotations

import numpy as np


def linear_regression_data(
    n_samples: int = 200,
    n_features: int = 2,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate y = X @ w + b + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=(n_features, 1))
    true_b = rng.normal()
    y = X @ true_w + true_b + rng.normal(scale=noise_std, size=(n_samples, 1))
    return X, y, true_w.ravel()


def binary_classification_data(
    n_per_class: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs for binary classification."""
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=[-1.5, -1.5], scale=0.8, size=(n_per_class, 2))
    class1 = rng.normal(loc=[1.5, 1.5], scale=0.8, size=(n_per_class, 2))
    X = np.vstack([class0, class1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def multiclass_blobs(
    n_per_class: int = 80,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian blobs for multiclass softmax regression."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, size=(n_classes, 2))
    X_list, y_list = [], []
    for c in range(n_classes):
        points = rng.normal(loc=centers[c], scale=0.7, size=(n_per_class, 2))
        X_list.append(points)
        y_list.append(np.full(n_per_class, c))
    return np.vstack(X_list), np.concatenate(y_list)


def spiral_data(
    n_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Spiral dataset that linear models cannot separate."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(scale=noise, size=n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def xor_data(n_per_quadrant: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """XOR-like dataset for testing nonlinear models."""
    rng = np.random.default_rng(seed)
    centers = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    labels = [0, 1, 1, 0]
    X_list, y_list = [], []
    for center, label in zip(centers, labels):
        points = rng.normal(loc=center, scale=0.3, size=(n_per_quadrant, 2))
        X_list.append(points)
        y_list.append(np.full(n_per_quadrant, label))
    return np.vstack(X_list), np.concatenate(y_list)
