"""Fine-tuning loop utilities (educational, numpy-friendly concepts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class FinetuneConfig:
    lr_head: float = 0.01
    lr_body: float = 0.001
    freeze_embeddings: bool = True
    n_epochs: int = 100


def run_finetune_epochs(
    n_epochs: int,
    train_fn: Callable[[int], float],
) -> list[float]:
    """Generic epoch loop returning per-epoch metric (e.g. accuracy)."""
    history = []
    for epoch in range(n_epochs):
        metric = train_fn(epoch)
        history.append(metric)
    return history


def describe_strategy(freeze_embeddings: bool, unfreeze_layers: int = 0) -> str:
    if freeze_embeddings and unfreeze_layers == 0:
        return "linear_probe"
    if freeze_embeddings:
        return f"partial_finetune_last_{unfreeze_layers}_layers"
    return "full_finetune"
