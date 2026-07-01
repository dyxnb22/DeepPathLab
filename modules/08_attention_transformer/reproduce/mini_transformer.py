#!/usr/bin/env python3
"""Mini transformer on copy problem for sequence modeling."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/07_lstm_gru"))

from copy_task import generate_copy_batch
from lib.plotting import module_output_dir, save_metric_curves


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.n_classes = vocab_size + 1
        self.embed = nn.Embedding(self.n_classes, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embed(x))
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.fc(h)


def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return loss[mask.reshape(-1)].mean()


def train_transformer(
    seq_len: int = 10,
    delay: int = 2,
    vocab_size: int = 4,
    n_epochs: int = 100,
) -> list[float]:
    model = MiniTransformer(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    val_accs = []

    for epoch in range(n_epochs):
        x, y, mask = generate_copy_batch(64, seq_len, delay, vocab_size, seed=epoch)
        y_safe = y.copy()
        y_safe[~mask] = 0
        x_t = torch.tensor(x, dtype=torch.long)
        y_t = torch.tensor(y_safe, dtype=torch.long)
        mask_t = torch.tensor(mask)

        optimizer.zero_grad()
        logits = model(x_t)
        loss = masked_ce(logits, y_t, mask_t)
        loss.backward()
        optimizer.step()

        x_v, y_v, m_v = generate_copy_batch(64, seq_len, delay, vocab_size, seed=500 + epoch)
        yv = y_v.copy()
        yv[~m_v] = 0
        with torch.no_grad():
            preds = model(torch.tensor(x_v, dtype=torch.long)).argmax(-1)
            vacc = ((preds == torch.tensor(yv)) & torch.tensor(m_v)).sum().float() / m_v.sum()
        val_accs.append(float(vacc))
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}: loss={loss.item():.3f}, val_acc={val_accs[-1]:.3f}")

    return val_accs


def main() -> None:
    accs = train_transformer()
    out = module_output_dir("08_attention_transformer") / "mini_transformer_copy.png"
    save_metric_curves({"Transformer": accs}, out, title="Mini Transformer on Copy Task")
    print(f"\nFinal val acc: {accs[-1]:.3f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
