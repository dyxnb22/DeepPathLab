"""Fine-tuning loop utilities (educational, numpy-friendly concepts).

Provides configuration and naming helpers for comparing fine-tuning strategies.
The actual training logic lives in reproduce/ and experiments/ scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class FinetuneConfig:
    """Hyperparameters for differential learning rates across body vs head."""

    lr_head: float = 0.01  # new classification head — typically larger
    lr_body: float = 0.001  # pretrained layers — typically smaller
    freeze_embeddings: bool = True
    n_epochs: int = 100


def run_finetune_epochs(
    n_epochs: int,
    train_fn: Callable[[int], float],
) -> list[float]:
    """Generic epoch loop returning per-epoch metric (e.g. accuracy).

    train_fn receives the epoch index and returns one scalar metric.
    """
    history = []
    for epoch in range(n_epochs):
        metric = train_fn(epoch)
        history.append(metric)
    return history


def describe_strategy(freeze_embeddings: bool, unfreeze_layers: int = 0) -> str:
    """Map freeze/unfreeze flags to a human-readable strategy label."""
    if freeze_embeddings and unfreeze_layers == 0:
        return "linear_probe"
    if freeze_embeddings:
        return f"partial_finetune_last_{unfreeze_layers}_layers"
    return "full_finetune"
