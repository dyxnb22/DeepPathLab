#!/usr/bin/env python3
"""Tiny masked language model demo with PyTorch."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from lib.plotting import module_output_dir, save_loss_curve


class TinyMLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=64, batch_first=True),
            num_layers=1,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.encoder(h)
        return self.fc(h)


def build_sentences() -> list[list[str]]:
    return [
        "the cat sits on the mat".split(),
        "the dog runs in the park".split(),
        "embeddings capture word meaning".split(),
        "transformers use self attention".split(),
        "king and queen are royalty".split(),
    ] * 20


def main() -> None:
    sents = build_sentences()
    vocab = sorted(set(w for s in sents for w in s))
    stoi = {w: i for i, w in enumerate(vocab)}
    mask_id = len(vocab)
    vocab_size = mask_id + 1

    pairs = []
    for sent in sents:
        ids = [stoi[w] for w in sent]
        for i in range(len(ids)):
            masked = ids.copy()
            target = masked[i]
            masked[i] = mask_id
            pairs.append((masked, i, target))

    model = TinyMLM(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(150):
        random.shuffle(pairs)
        epoch_loss = 0.0
        for masked, pos, target in pairs:
            x = torch.tensor([masked])
            logits = model(x)[0, pos]
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([target]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(pairs))
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}: loss={losses[-1]:.4f}")

    # Demo prediction
    demo = "the cat sits on the mat".split()
    ids = [stoi[w] for w in demo]
    masked = ids.copy()
    masked[2] = mask_id
    with torch.no_grad():
        logits = model(torch.tensor([masked]))[0, 2]
        pred = vocab[logits.argmax().item()]
    print(f"\nMLM demo: 'the cat [MASK] on the mat' -> '{pred}'")

    out = module_output_dir("11_nlp_pretraining") / "mlm_loss.png"
    save_loss_curve(losses, out, title="Tiny MLM Training Loss")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
