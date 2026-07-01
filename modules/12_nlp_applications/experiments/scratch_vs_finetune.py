#!/usr/bin/env python3
"""Compare training from scratch vs using pretrained skip-gram embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/11_nlp_pretraining/from_scratch"))
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/from_scratch"))
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/reproduce"))

from bow_classifier import load_tsv, tokenize
from skipgram import build_vocab as sg_vocab, generate_pairs, load_corpus, train_skipgram
from text_classifier import train_classifier


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "corpus" / "tiny_sentiment.tsv"
    texts, labels = load_tsv(data_path)

    _, scratch_acc = train_classifier(texts, labels, n_epochs=250, lr=0.01, pretrained_emb=None)
    print(f"From scratch: {scratch_acc:.3f}")

    corpus = Path(__file__).resolve().parents[2] / "11_nlp_pretraining" / "corpus" / "tiny_corpus.txt"
    tokens = load_corpus(corpus)
    stoi, words = sg_vocab(tokens)
    pairs = generate_pairs(tokens, stoi, window=2)
    emb = train_skipgram(pairs, len(words), dim=16, n_epochs=300)

    # Map sentiment vocab to pretrained where possible
    clf_vocab = {"<pad>": 0}
    for text in texts:
        for tok in tokenize(text):
            if tok not in clf_vocab:
                clf_vocab[tok] = len(clf_vocab)

    pretrained = torch.zeros(len(clf_vocab), 16)
    for word, idx in clf_vocab.items():
        if word in stoi:
            pretrained[idx] = torch.tensor(emb[stoi[word]])

    _, finetune_acc = train_classifier(texts, labels, n_epochs=250, lr=0.01, pretrained_emb=pretrained)
    print(f"With pretrained init: {finetune_acc:.3f}")


if __name__ == "__main__":
    main()
