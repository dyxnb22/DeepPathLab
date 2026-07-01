"""从零实现 Skip-gram Word2Vec（简化版，numpy）。

训练流程：分词 → 建词表 → 生成 (中心词, 上下文) 对 → softmax 交叉熵更新。
使用全词表 softmax（未做负采样），适合微型语料教学。

两套矩阵：W_in（词向量，训练后作为嵌入）与 W_out（输出投影）。
公式与性质参见 notes.md。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    """小写化并提取英文字母词元。"""
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(tokens: list[str], min_count: int = 1) -> tuple[dict[str, int], list[str]]:
    """统计词频，过滤低频词，返回 stoi 映射与词列表。"""
    counts = Counter(tokens)
    words = [w for w, c in counts.items() if c >= min_count]
    stoi = {w: i for i, w in enumerate(words)}
    return stoi, words


def generate_pairs(tokens: list[str], stoi: dict[str, int], window: int = 2) -> list[tuple[int, int]]:
    """滑动窗口生成 (中心词 id, 上下文词 id) 训练对。

    window=2 表示中心词左右各 2 个位置内的词作为正样本。
    """
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
    """Skip-gram 训练：对每个 (center, context) 对做 softmax 梯度下降。

    Returns:
        W_in：shape (vocab_size, dim)，即学得的词向量矩阵。
    """
    rng = np.random.default_rng(seed)
    W_in = rng.normal(scale=0.1, size=(vocab_size, dim))
    W_out = rng.normal(scale=0.1, size=(vocab_size, dim))

    for _ in range(n_epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            h = W_in[center]
            logits = W_out @ h
            # 数值稳定：减去最大值再 softmax
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            # 交叉熵对 logits 的梯度：probs - one_hot(context)
            grad_logits = probs.copy()
            grad_logits[context] -= 1.0
            W_out -= lr * np.outer(grad_logits, h)
            W_in[center] -= lr * (W_out.T @ grad_logits)

    return W_in


def nearest(word: str, embeddings: np.ndarray, stoi: dict[str, int], itos: list[str], k: int = 3) -> list[str]:
    """按余弦相似度返回 top-k 最近邻（排除自身）。"""
    if word not in stoi:
        return []
    vec = embeddings[stoi[word]]
    sims = embeddings @ vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-8)
    order = np.argsort(-sims)
    return [itos[i] for i in order[1 : k + 1] if itos[i] != word]


def load_corpus(path: Path) -> list[str]:
    """读取语料文件并分词。"""
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
