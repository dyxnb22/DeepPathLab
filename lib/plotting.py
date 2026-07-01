"""Simple plotting helpers for training curves and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_curve(
    losses: list[float],
    output_path: Path | str,
    title: str = "Training Loss",
    xlabel: str = "Step",
    ylabel: str = "Loss",
) -> Path:
    """Save a loss curve to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_metric_curves(
    metrics: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Training Metrics",
    xlabel: str = "Epoch",
) -> Path:
    """Save multiple metric curves on one plot."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in metrics.items():
        ax.plot(values, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def module_output_dir(module_name: str, repo_root: Path | None = None) -> Path:
    """Return the standard output directory for a module."""
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "modules" / module_name / "outputs"
