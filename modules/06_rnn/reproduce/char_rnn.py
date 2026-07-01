#!/usr/bin/env python3
"""Character-level RNN text generator (PyTorch)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lib.plotting import module_output_dir, save_loss_curve


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, n_layers: int = 1) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(x)
        out, h = self.rnn(emb, h)
        logits = self.fc(out)
        return logits, h


def build_corpus() -> str:
    return (
        "deep learning is about building representations from data. "
        "recurrent networks process sequences one step at a time. "
        "each hidden state summarizes the past. "
    ) * 20


def encode(text: str) -> tuple[list[int], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = [stoi[c] for c in text]
    return ids, stoi, itos


def make_batches(ids: list[int], seq_len: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = (len(ids) - 1) // batch_size
    data = torch.tensor(ids[: n * batch_size + 1])
    x = data[:-1].view(batch_size, n)
    y = data[1:].view(batch_size, n)
    return x[:, :seq_len], y[:, :seq_len]


def train_char_rnn(n_epochs: int = 200, seq_len: int = 32) -> list[float]:
    text = build_corpus()
    ids, stoi, itos = encode(text)
    vocab_size = len(stoi)
    model = CharRNN(vocab_size, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(n_epochs):
        x, y = make_batches(ids, seq_len, batch_size=4)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: loss={losses[-1]:.4f}")

    # Generate sample
    model.eval()
    with torch.no_grad():
        h = None
        seed_char = "d"
        idx = stoi[seed_char]
        generated = seed_char
        for _ in range(60):
            inp = torch.tensor([[idx]])
            logits, h = model(inp, h)
            idx = int(logits[0, -1].argmax())
            generated += itos[idx]
    print(f"\nGenerated sample:\n{generated[:80]}...")

    out = module_output_dir("06_rnn") / "char_rnn_loss.png"
    save_loss_curve(losses, out, title="Char-RNN Training Loss")
    return losses


def main() -> None:
    train_char_rnn()


if __name__ == "__main__":
    main()
