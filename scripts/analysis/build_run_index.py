#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.analysis.io import build_run_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build logs/run_index.csv from experiment logs.")
    parser.add_argument("--root", default="logs", help="Root logs directory.")
    parser.add_argument("--output", default="logs/run_index.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = build_run_index(root=args.root, output_path=args.output)
    print(f"Wrote {len(df)} rows to {Path(args.output).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
