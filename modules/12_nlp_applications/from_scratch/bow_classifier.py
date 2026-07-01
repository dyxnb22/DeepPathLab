"""Bag-of-words logistic regression classifier from scratch."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


def load_tsv(path: Path) -> tuple[list[str], list[int]]:
    labels, texts = [], []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        label, text = line.split(" ", 1)
        labels.append(1 if label == "positive" else 0)
        texts.append(text)
    return texts, labels


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_bow(texts: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    vocab: dict[str, int] = {}
    for text in texts:
        for tok in set(tokenize(text)):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        for tok in tokenize(text):
            if tok in vocab:
                X[i, vocab[tok]] += 1
    return X, vocab


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def train_bow_classifier(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_epochs: int = 300,
) -> np.ndarray:
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(n_epochs):
        probs = sigmoid(X @ w + b)
        grad_w = (X.T @ (probs - y)) / n
        grad_b = float(np.sum(probs - y)) / n
        w -= lr * grad_w
        b -= lr * grad_b
    return w


def predict(X: np.ndarray, w: np.ndarray, b: float = 0.0) -> np.ndarray:
    return (sigmoid(X @ w + b) >= 0.5).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "corpus" / "tiny_sentiment.tsv"
    texts, labels = load_tsv(path)
    X, vocab = build_bow(texts)
    y = np.array(labels, dtype=float)
    w = train_bow_classifier(X, y)
    preds = predict(X, w)
    print(f"BoW classifier accuracy: {accuracy(y, preds):.3f} ({len(vocab)} features)")
