#!/usr/bin/env python3
"""Visualize attention weights from mini transformer."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/08_attention_transformer/reproduce"))

from lib.plotting import module_output_dir
from mini_transformer import MiniTransformer


class TransformerWithAttnWeights(MiniTransformer):
    """Wrapper to extract attention weights from first encoder layer."""

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h = self.pos(self.embed(x))
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            # Access first layer self-attention
            layer = self.encoder.layers[0]
            h_norm = layer.norm1(h)
            attn_out, weights = layer.self_attn(
                h_norm, h_norm, h_norm, attn_mask=mask, need_weights=True, average_attn_weights=True
            )
            return weights[0]  # (seq, seq)


def main() -> None:
    vocab_size = 4
    seq_len, delay = 10, 2
    model = TransformerWithAttnWeights(vocab_size)

    # Construct one copy example: symbols at start, recall at end
    x = torch.full((1, seq_len), vocab_size, dtype=torch.long)  # blank
    symbols = torch.tensor([0, 1, 2, 3])
    x[0, :4] = symbols
    # recall positions 6-9 should attend to 0-3

    weights = model.get_attention_map(x)
    w = weights.numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(w, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("Self-Attention Weights (layer 0)")
    fig.colorbar(im, ax=ax)
    out = module_output_dir("08_attention_transformer") / "attention_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved attention heatmap to {out}")
    print("Inspect: recall positions (rows 6-9) should attend to early symbol positions (cols 0-3)")


if __name__ == "__main__":
    main()
