#!/usr/bin/env python3
"""PyTorch text classifier with learned embeddings."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/from_scratch"))

from bow_classifier import load_tsv, tokenize


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x).mean(dim=1)
        return self.fc(emb).squeeze(-1)


def encode(texts: list[str], stoi: dict[str, int], max_len: int = 12) -> torch.Tensor:
    ids = []
    for text in texts:
        toks = tokenize(text)[:max_len]
        row = [stoi.get(t, 0) for t in toks]
        row += [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0}
    for text in texts:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def train_classifier(texts: list[str], labels: list[int], n_epochs: int = 200, lr: float = 0.01, pretrained_emb: torch.Tensor | None = None) -> tuple[TextClassifier, float]:
    stoi = build_vocab(texts)
    X = encode(texts, stoi)
    y = torch.tensor(labels, dtype=torch.float32)
    model = TextClassifier(len(stoi))
    if pretrained_emb is not None and pretrained_emb.shape[0] <= len(stoi):
        with torch.no_grad():
            model.embed.weight[: pretrained_emb.shape[0]] = pretrained_emb
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) >= 0.5).float()
        acc = float((preds == y).float().mean())
    return model, acc


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "corpus" / "tiny_sentiment.tsv"
    texts, labels = load_tsv(path)
    _, acc = train_classifier(texts, labels)
    print(f"PyTorch text classifier accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
