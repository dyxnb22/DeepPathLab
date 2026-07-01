"""Skip-gram word2vec from scratch (simplified)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict[str, int], list[str]]:
    counts = Counter(tokens)
    words = [w for w, c in counts.items() if c >= min_count]
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, words


def generate_pairs(tokens: list[str], stoi: dict[str, int], window: int = 2) -> list[tuple[int, int]]:
    pairs = []
    for i, word in enumerate(tokens):
        if word not in stoi:
            continue
        center = stoi[word]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i == j:
                continue
            ctx = tokens[j]
            if ctx in stoi:
                pairs.append((center, stoi[ctx]))
    return pairs


def train_skipgram(
    pairs: list[tuple[int, int]],
    vocab_size: int,
    dim: int = 16,
    lr: float = 0.05,
    n_epochs: int = 200,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W_in = rng.normal(scale=0.1, size=(vocab_size, dim))
    W_out = rng.normal(scale=0.1, size=(vocab_size, dim))

    for _ in range(n_epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            h = W_in[center]
            logits = W_out @ h
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            grad_logits = probs.copy()
            grad_logits[context] -= 1.0
            W_out -= lr * np.outer(grad_logits, h)
            W_in[center] -= lr * (W_out.T @ grad_logits)

    return W_in


def nearest(word: str, embeddings: np.ndarray, stoi: dict[str, int], itos: list[str], k: int = 3) -> list[str]:
    if word not in stoi:
        return []
    vec = embeddings[stoi[word]]
    sims = embeddings @ vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-8)
    order = np.argsort(-sims)
    return [itos[i] for i in order[1 : k + 1] if itos[i] != word]


def load_corpus(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return tokenize(text)


if __name__ == "__main__":
    corpus_path = Path(__file__).resolve().parents[1] / "corpus" / "tiny_corpus.txt"
    tokens = load_corpus(corpus_path)
    stoi, words = build_vocab(tokens)
    pairs = generate_pairs(tokens, stoi, window=2)
    emb = train_skipgram(pairs, len(words), dim=16, n_epochs=300)
    for w in ["king", "queen", "word", "embeddings"]:
        print(f"Nearest to '{w}': {nearest(w, emb, stoi, words)}")
