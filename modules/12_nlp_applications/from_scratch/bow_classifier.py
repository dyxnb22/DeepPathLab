"""从零实现词袋（BoW）+ Logistic Regression 情感分类器（numpy）。

流水线：读取 TSV → 分词建词表 → 构建 BoW 矩阵 → 梯度下降训练 → 预测。
作为 NLP 下游任务的最简强基线，不依赖 PyTorch。

损失为二元交叉熵；sigmoid 内 clip 防止 exp 溢出。公式参见 notes.md。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


def load_tsv(path: Path) -> tuple[list[str], list[int]]:
    """读取 `label text` 格式 TSV，label 为 positive/negative。"""
    labels, texts = [], []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        label, text = line.split(" ", 1)
        labels.append(1 if label == "positive" else 0)
        texts.append(text)
    return texts, labels


def tokenize(text: str) -> list[str]:
    """小写化并提取英文字母词元。"""
    return re.findall(r"[a-z]+", text.lower())


def build_bow(texts: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """构建词袋矩阵 X shape (n_samples, vocab_size)，值为词频计数。

    词表从训练文本中提取；推理时 OOV 词被忽略。
    """
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
    """数值稳定的 sigmoid：clip 输入避免 exp 溢出。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def train_bow_classifier(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_epochs: int = 300,
) -> np.ndarray:
    """批量梯度下降训练 logistic regression，返回权重 w（偏置 b 固定为 0）。

    梯度：∂L/∂w = Xᵀ(probs - y) / N
    """
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
    """阈值 0.5 二分类预测。"""
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
