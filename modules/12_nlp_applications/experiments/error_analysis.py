#!/usr/bin/env python3
"""Error analysis on sentiment classifier predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/from_scratch"))
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/reproduce"))

from bow_classifier import load_tsv
from text_classifier import build_vocab, encode, train_classifier


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "corpus" / "tiny_sentiment.tsv"
    texts, labels = load_tsv(path)
    model, acc = train_classifier(texts, labels, n_epochs=300)
    stoi = build_vocab(texts)
    X = encode(texts, stoi)
    y = torch.tensor(labels, dtype=torch.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(X))
        preds = (probs >= 0.5).float()

    print(f"Overall accuracy: {acc:.3f}\n")
    print("Misclassified examples:")
    count = 0
    for i, (text, true_l, pred, p) in enumerate(zip(texts, labels, preds, probs)):
        if int(pred) != true_l:
            label_name = "positive" if true_l else "negative"
            pred_name = "positive" if pred else "negative"
            print(f"  [{label_name} -> {pred_name}] (p={p:.2f}) {text}")
            count += 1
    if count == 0:
        print("  (none — perfect on training set)")
    print(f"\nTotal errors: {count}/{len(texts)}")


if __name__ == "__main__":
    main()
