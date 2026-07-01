#!/usr/bin/env python3
"""Measure gradient norm vs sequence length in vanilla RNN."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/06_rnn/from_scratch"))

from lib.plotting import module_output_dir, save_metric_curves
from rnn import rnn_backward_full, rnn_forward, sequence_copy_task


def grad_norm_vs_length(lengths: list[int]) -> dict[str, list[float]]:
    vocab_size = 4
    hidden_dim = 32
    rng = np.random.default_rng(0)
    results: dict[str, list[float]] = {
        "seq_len": [],
        "W_hh_norm": [],
        "dh0_norm": [],
    }

    for seq_len in lengths:
        W_xh = rng.normal(scale=0.1, size=(hidden_dim, vocab_size))
        W_hh = np.eye(hidden_dim) * 0.5  # controlled recurrent weights for vanishing demo
        b_h = np.zeros(hidden_dim)
        W_hy = rng.normal(scale=0.1, size=(vocab_size, hidden_dim))
        b_y = np.zeros(vocab_size)

        x_seq, targets = sequence_copy_task(seq_len, vocab_size)
        hidden_states, outputs, _ = rnn_forward(x_seq, W_xh, W_hh, b_h, W_hy, b_y)
        grads, loss, dh0_norm = rnn_backward_full(
            x_seq, hidden_states, outputs, targets, W_xh, W_hh, W_hy
        )

        results["seq_len"].append(seq_len)
        results["W_hh_norm"].append(float(np.linalg.norm(grads["W_hh"])))
        results["dh0_norm"].append(dh0_norm)
        print(
            f"seq_len={seq_len:3d}, loss={loss:.4f}, "
            f"||dW_hh||={results['W_hh_norm'][-1]:.6f}, |dh_0|={dh0_norm:.6f}"
        )

    return results


def main() -> None:
    lengths = [2, 4, 8, 16, 32, 64]
    results = grad_norm_vs_length(lengths)
    out = module_output_dir("06_rnn") / "grad_vs_seq_len.png"
    x_labels = [str(l) for l in results["seq_len"]]
    save_metric_curves(
        {"dW_hh norm": results["W_hh_norm"], "|dh_0| after BPTT": results["dh0_norm"]},
        out,
        title="RNN Gradient Norm vs Sequence Length",
        xlabel="Epoch",
    )
    print(f"\nSaved plot to {out}")
    print("Observation: |dh_0| shrinks as sequence length grows (vanishing gradients).")


if __name__ == "__main__":
    main()
