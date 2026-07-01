#!/usr/bin/env python3
"""Run a quick sanity check for each DeepPath Lab module.

Each check runs the module's key script as a subprocess with a timeout.
Prints PASS/FAIL per module and exits with code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# module_id -> (display name, script path relative to repo root, timeout seconds)
MODULE_CHECKS: dict[str, tuple[str, str, int]] = {
    "01": ("01 Preliminaries & Autograd", "modules/01_preliminaries_autograd/experiments/gradient_check.py", 30),
    "02": ("02 Linear Models", "modules/02_linear_models/from_scratch/linear_regression.py", 30),
    "03": ("03 Multilayer Perceptrons", "modules/03_mlp/from_scratch/mlp_numpy.py", 60),
    "04": ("04 Convolutional Neural Networks", "modules/04_cnn/from_scratch/conv2d.py", 60),
    "05": ("05 Modern CNN", "modules/05_modern_cnn/from_scratch/residual_block.py", 30),
    "06": ("06 RNN", "modules/06_rnn/from_scratch/rnn.py", 60),
    "07": ("07 LSTM & GRU", "modules/07_lstm_gru/from_scratch/lstm_cell.py", 30),
    "08": ("08 Attention & Transformer", "modules/08_attention_transformer/from_scratch/attention.py", 30),
    "09": ("09 Optimization", "modules/09_optimization/from_scratch/optimizers.py", 30),
    "10": ("10 Computer Vision Applications", "modules/10_computer_vision_applications/from_scratch/augmentation.py", 60),
    "11": ("11 NLP Pretraining", "modules/11_nlp_pretraining/from_scratch/skipgram.py", 60),
    "12": ("12 NLP Applications", "modules/12_nlp_applications/from_scratch/bow_classifier.py", 30),
    "13": ("13 NLP Fine-Tuning", "modules/13_nlp_fine_tuning/experiments/freeze_vs_finetune.py", 120),
    "14": ("14 Recommender Systems", "modules/14_recommender_systems/from_scratch/matrix_factorization.py", 60),
    "15": ("15 Reinforcement Learning", "modules/15_reinforcement_learning/from_scratch/q_learning.py", 30),
}


def run_check(module_id: str, name: str, script: str, timeout: int) -> bool:
    path = REPO_ROOT / script
    if not path.exists():
        print(f"  FAIL  {name}: script not found ({script})")
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  FAIL  {name}: timeout after {timeout}s")
        return False
    if result.returncode == 0:
        print(f"  PASS  {name}")
        return True
    err = (result.stderr or result.stdout or "").strip().splitlines()
    snippet = err[-1] if err else f"exit code {result.returncode}"
    print(f"  FAIL  {name}: {snippet}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DeepPath Lab modules")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="Run checks for specific module IDs (e.g. 13 14 15)",
    )
    args = parser.parse_args()

    ids = sorted(MODULE_CHECKS.keys(), key=lambda x: int(x))
    if args.only:
        requested = {m.zfill(2) if m.isdigit() else m for m in args.only}
        ids = [i for i in ids if i in requested or i.lstrip("0") in requested]

    print(f"DeepPath Lab verify ({len(ids)} modules)\n")
    results = [run_check(mid, *MODULE_CHECKS[mid]) for mid in ids]

    passed = sum(results)
    print(f"\n{passed}/{len(ids)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
