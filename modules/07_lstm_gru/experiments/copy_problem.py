#!/usr/bin/env python3
"""Compare RNN/LSTM/GRU on easy and hard copy problem settings."""

from __future__ import annotations

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


class CopyRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 64, cell: str = "rnn", num_layers: int = 1) -> None:
        super().__init__()
        self.n_classes = vocab_size + 1
        self.input_proj = nn.Linear(self.n_classes, hidden_dim)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(x, self.n_classes).float()
        h = self.input_proj(one_hot)
        out, _ = self.rnn(h)
        return self.fc(out)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    return loss[mask.reshape(-1)].mean()


def train_copy_model(
    cell: str,
    seq_len: int,
    delay: int,
    vocab_size: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 1,
    n_epochs: int = 150,
    batch_size: int = 64,
    lr: float = 0.01,
) -> list[float]:
    torch.manual_seed(42)
    model = CopyRNN(vocab_size, hidden_dim, cell, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_accs = []

    for epoch in range(n_epochs):
        x, y, mask = generate_copy_batch(batch_size, seq_len, delay, vocab_size, seed=epoch)
        y_safe = y.copy()
        y_safe[~mask] = 0
        x_t = torch.tensor(x, dtype=torch.long)
        y_t = torch.tensor(y_safe, dtype=torch.long)
        mask_t = torch.tensor(mask)

        optimizer.zero_grad()
        logits = model(x_t)
        loss = masked_cross_entropy(logits, y_t, mask_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        x_v, y_v, m_v = generate_copy_batch(batch_size, seq_len, delay, vocab_size, seed=999 + epoch)
        yv_safe = y_v.copy()
        yv_safe[~m_v] = 0
        with torch.no_grad():
            logits_v = model(torch.tensor(x_v, dtype=torch.long))
            preds = logits_v.argmax(-1)
            vacc = ((preds == torch.tensor(yv_safe)) & torch.tensor(m_v)).sum().float() / m_v.sum()
        val_accs.append(float(vacc))

        if (epoch + 1) % 50 == 0:
            print(f"  [{cell}] epoch {epoch + 1}: loss={loss.item():.3f}, val_acc={val_accs[-1]:.3f}")

    return val_accs


def run_setting(name: str, seq_len: int, delay: int) -> dict[str, list[float]]:
    print(f"\n=== {name} (seq_len={seq_len}, delay={delay}) ===")
    curves = {}
    for cell, label in [("rnn", "RNN"), ("lstm", "LSTM"), ("gru", "GRU")]:
        curves[label] = train_copy_model(cell, seq_len, delay)
    return curves


def main() -> None:
    easy = run_setting("Easy copy", seq_len=10, delay=2)   # L=4
    hard = run_setting("Hard copy", seq_len=30, delay=10)  # L=10

    out_easy = module_output_dir("07_lstm_gru") / "copy_easy.png"
    out_hard = module_output_dir("07_lstm_gru") / "copy_hard.png"
    save_metric_curves(easy, out_easy, title="Copy Problem (delay=2)", xlabel="Epoch")
    save_metric_curves(hard, out_hard, title="Copy Problem (delay=10)", xlabel="Epoch")

    print("\nFinal val accuracy (easy):")
    for k, v in easy.items():
        print(f"  {k}: {v[-1]:.3f}")
    print("Final val accuracy (hard):")
    for k, v in hard.items():
        print(f"  {k}: {v[-1]:.3f}")


if __name__ == "__main__":
    main()
