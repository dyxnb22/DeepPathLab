#!/usr/bin/env python3
"""Fine-tune text classifier: frozen vs full fine-tune vs scratch."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modules/11_nlp_pretraining/from_scratch"))
sys.path.insert(0, str(REPO_ROOT / "modules/12_nlp_applications/from_scratch"))

from bow_classifier import load_tsv, tokenize
from lib.plotting import module_output_dir, save_metric_curves
from skipgram import build_vocab as sg_vocab, generate_pairs, load_corpus, train_skipgram


class FinetuneClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x).mean(dim=1)).squeeze(-1)


def encode(texts: list[str], stoi: dict[str, int], max_len: int = 10) -> torch.Tensor:
    rows = []
    for text in texts:
        toks = tokenize(text)[:max_len]
        row = [stoi.get(t, 0) for t in toks] + [0] * (max_len - len(toks))
        rows.append(row[:max_len])
    return torch.tensor(rows, dtype=torch.long)


def build_clf_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0}
    for text in texts:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def load_pretrained_matrix(stoi: dict[str, int], sg_stoi: dict[str, int], sg_emb) -> torch.Tensor:
    dim = sg_emb.shape[1]
    mat = torch.randn(len(stoi), dim) * 0.01
    for word, idx in stoi.items():
        if word in sg_stoi:
            mat[idx] = torch.tensor(sg_emb[sg_stoi[word]])
    return mat


def train_model(
    model: FinetuneClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int,
    lr: float,
    freeze_embed: bool,
) -> list[float]:
    for p in model.embed.parameters():
        p.requires_grad = not freeze_embed
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    accs = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        accs.append(float(((torch.sigmoid(logits) >= 0.5).float() == y).float().mean()))
    return accs


def main() -> None:
    train_path = Path(__file__).resolve().parents[1] / "corpus" / "sentiment_train.tsv"
    val_path = Path(__file__).resolve().parents[1] / "corpus" / "sentiment_val.tsv"
    train_texts, train_labels = load_tsv(train_path)
    val_texts, val_labels = load_tsv(val_path)
    all_texts = train_texts + val_texts
    stoi = build_clf_vocab(all_texts)

    corpus = REPO_ROOT / "modules/11_nlp_pretraining/corpus/tiny_corpus.txt"
    sg_tokens = load_corpus(corpus)
    sg_stoi, _ = sg_vocab(sg_tokens)
    sg_emb = train_skipgram(generate_pairs(sg_tokens, sg_stoi, 2), len(sg_stoi), dim=16, n_epochs=200)
    pretrained = load_pretrained_matrix(stoi, sg_stoi, sg_emb)

    X_train = encode(train_texts, stoi)
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    X_val = encode(val_texts, stoi)
    y_val = torch.tensor(val_labels, dtype=torch.float32)

    strategies = {
        "scratch": (False, None),
        "frozen_embed": (True, pretrained),
        "full_finetune": (False, pretrained),
    }
    curves: dict[str, list[float]] = {}

    for name, (freeze, init_emb) in strategies.items():
        model = FinetuneClassifier(len(stoi))
        if init_emb is not None:
            with torch.no_grad():
                model.embed.weight.copy_(init_emb)
        train_model(model, X_train, y_train, n_epochs=150, lr=0.01, freeze_embed=(name == "frozen_embed"))
        with torch.no_grad():
            val_acc = float(((torch.sigmoid(model(X_val)) >= 0.5).float() == y_val).float().mean())
        curves[name] = [val_acc]  # store final val for bar-style; also train history
        # re-run for training curve on val each 10 epochs - simplified: use train acc history
        m2 = FinetuneClassifier(len(stoi))
        if init_emb is not None:
            with torch.no_grad():
                m2.embed.weight.copy_(init_emb)
        hist = train_model(m2, X_train, y_train, 150, 0.01, name == "frozen_embed")
        curves[name] = hist
        with torch.no_grad():
            vacc = float(((torch.sigmoid(m2(X_val)) >= 0.5).float() == y_val).float().mean())
        print(f"  {name}: final_train_acc={hist[-1]:.3f}, val_acc={vacc:.3f}")

    out = module_output_dir("13_nlp_fine_tuning") / "finetune_strategies.png"
    save_metric_curves(curves, out, title="Fine-tuning Strategies (Train Acc)", xlabel="Epoch")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
