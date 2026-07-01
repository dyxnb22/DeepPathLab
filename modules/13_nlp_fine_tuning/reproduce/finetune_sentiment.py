#!/usr/bin/env python3
"""Entry point for NLP fine-tuning benchmark."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments" / "freeze_vs_finetune.py"))
