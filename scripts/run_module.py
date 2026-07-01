#!/usr/bin/env python3
"""Run a module's primary script by ID.

Examples:
    python3 scripts/run_module.py 01
    python3 scripts/run_module.py 08 --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_all import MODULE_CHECKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DeepPath Lab module's key script")
    parser.add_argument("module_id", nargs="?", help="Module ID, e.g. 01 or 8")
    parser.add_argument("--list", action="store_true", help="List all modules and scripts")
    args = parser.parse_args()

    if args.list or not args.module_id:
        print("Available modules:\n")
        for mid in sorted(MODULE_CHECKS, key=lambda x: int(x)):
            name, script, _ = MODULE_CHECKS[mid]
            print(f"  {mid}  {name}")
            print(f"      python3 {script}\n")
        return 0

    mid = args.module_id.zfill(2) if args.module_id.isdigit() else args.module_id
    if mid not in MODULE_CHECKS:
        print(f"Unknown module: {args.module_id}. Use --list.", file=sys.stderr)
        return 1

    name, script, _ = MODULE_CHECKS[mid]
    path = REPO_ROOT / script
    print(f"Running {name}\n  {script}\n")
    return subprocess.call([sys.executable, str(path)], cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
