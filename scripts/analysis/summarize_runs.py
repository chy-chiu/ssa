#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.analysis.io import discover_logs
from ssa.analysis.metrics import get_summary_df
from ssa.common import ExperimentLog

_PATH_LAYOUT_RE = re.compile(r"logs/(?P<study>[^/]+)/(?P<variant>[^/]+)/(?P<base>[^/]+)\.log$")
_RUN_RE = re.compile(r"(?P<run_name>.+)_(?P<replicate_id>\d+)$")


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _parse_path(path: Path) -> dict:
    match = _PATH_LAYOUT_RE.search(path.as_posix())
    if not match:
        return {"study": "unknown", "variant": "unknown", "run_name": path.stem, "replicate_id": -1}
    base = match.group("base")
    run_match = _RUN_RE.match(base)
    return {
        "study": match.group("study"),
        "variant": match.group("variant"),
        "run_name": run_match.group("run_name") if run_match else base,
        "replicate_id": int(run_match.group("replicate_id")) if run_match else -1,
    }


def _select_logs(root: str, study: str | None, variant: str | None, glob_pat: str, tag: str | None) -> List[Path]:
    logs = discover_logs(root=root, study=study, variant=variant)
    selected = []
    for path in logs:
        if glob_pat and not fnmatch.fnmatch(path.name, glob_pat):
            continue
        if tag and tag not in path.as_posix():
            continue
        selected.append(path)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize runs into agent-level and variant-level tables.")
    parser.add_argument("--root", default="logs")
    parser.add_argument("--study", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--glob", dest="glob_pat", default="*.log")
    parser.add_argument("--tag", default=None, help="Substring filter against full log path.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    paths = _select_logs(args.root, args.study, args.variant, args.glob_pat, args.tag)
    if not paths:
        print("No matching logs found.")
        return 0

    frames = []
    for path in paths:
        try:
            exp_log = ExperimentLog.load(str(path))
        except Exception:
            continue
        meta = _parse_path(path)
        frame = get_summary_df(exp_log, fp=path.as_posix())
        frame["study"] = meta["study"]
        frame["variant"] = meta["variant"]
        frame["run_name"] = meta["run_name"]
        frame["replicate_id"] = meta["replicate_id"]
        frames.append(frame)

    if not frames:
        print("No readable logs found.")
        return 0

    agent_df = pd.concat(frames, ignore_index=True)
    variant_df = (
        agent_df.groupby(["study", "variant", "atype"])
        .agg(
            n_agents=("agent_id", "count"),
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            train_p_mean=("train_p", "mean"),
            winrate_mean=("winrate", "mean"),
            recovery_mean=("recovery", "mean"),
        )
        .reset_index()
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.study and args.variant:
        out_dir = Path(args.root) / args.study / args.variant / "analysis"
    elif args.study:
        out_dir = Path(args.root) / args.study / "analysis"
    else:
        out_dir = Path(args.root) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_csv = out_dir / "agent_summary.csv"
    variant_csv = out_dir / "variant_summary.csv"
    agent_md = out_dir / "agent_summary.md"
    variant_md = out_dir / "variant_summary.md"

    agent_df.to_csv(agent_csv, index=False)
    variant_df.to_csv(variant_csv, index=False)
    agent_md.write_text(_to_markdown(agent_df), encoding="utf-8")
    variant_md.write_text(_to_markdown(variant_df), encoding="utf-8")

    print(f"Wrote {len(agent_df)} agent rows -> {agent_csv.as_posix()}")
    print(f"Wrote {len(variant_df)} variant rows -> {variant_csv.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
