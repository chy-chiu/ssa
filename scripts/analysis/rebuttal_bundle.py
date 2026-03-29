#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.analysis.io import build_run_index
from ssa.analysis.rebuttal import evaluate_rebuttal_claims


def _to_markdown(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact rebuttal bundle tables.")
    parser.add_argument("--root", default="logs")
    parser.add_argument("--output-dir", default="logs/rebuttal_bundle")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df = build_run_index(root=args.root, output_path=out_dir / "run_index.csv")
    claim_df = evaluate_rebuttal_claims(run_df)
    compact = run_df[
        [
            "study",
            "variant",
            "run_name",
            "replicate_id",
            "mean_winning_bid_ratio",
            "train_rate",
            "ssa_minus_control_reward",
        ]
    ] if len(run_df) else run_df

    compact.to_csv(out_dir / "compact_run_table.csv", index=False)
    claim_df.to_csv(out_dir / "directional_claims.csv", index=False)
    (out_dir / "compact_run_table.md").write_text(_to_markdown(compact), encoding="utf-8")
    (out_dir / "directional_claims.md").write_text(_to_markdown(claim_df), encoding="utf-8")

    print(f"Wrote rebuttal bundle to {out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
