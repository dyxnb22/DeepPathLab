#!/usr/bin/env python3
"""Visualize word embeddings with nearest neighbors."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/11_nlp_pretraining/from_scratch"))

from lib.plotting import module_output_dir
from skipgram import build_vocab, generate_pairs, load_corpus, nearest, train_skipgram


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def main() -> None:
    corpus = Path(__file__).resolve().parents[1] / "corpus" / "tiny_corpus.txt"
    tokens = load_corpus(corpus)
    stoi, words = build_vocab(tokens)
    pairs = generate_pairs(tokens, stoi, window=2)
    emb = train_skipgram(pairs, len(words), dim=16, n_epochs=400)

    coords = pca_2d(emb)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=30)
    for i, w in enumerate(words):
        ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=8)

    highlight = ["king", "queen", "man", "woman", "word", "embeddings"]
    for w in highlight:
        if w in stoi:
            nn = nearest(w, emb, stoi, words, k=2)
            print(f"  {w} -> {nn}")

    out = module_output_dir("11_nlp_pretraining") / "embedding_pca.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    ax.set_title("Skip-gram Embeddings (PCA)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
