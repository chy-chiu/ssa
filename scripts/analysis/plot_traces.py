#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
from collections import defaultdict
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssa.analysis.io import discover_logs
from ssa.common import ExperimentLog


def _collect_traces(exp_log: ExperimentLog) -> Dict[str, List[float]]:
    winning_bid_ratio = []
    training_rate = []
    mean_reward = []
    n_agents = max(len(exp_log.agent_ids), 1)

    for hx in exp_log.history:
        ratios = []
        for job_id, winning_price in hx.winning_prices.items():
            base = hx.base_prices.get(job_id, 0.0)
            if base > 0:
                ratios.append(float(winning_price) / float(base))
        winning_bid_ratio.append(float(np.mean(ratios)) if ratios else 0.0)
        training_rate.append(float(sum(1 for action in hx.agent_actions if action.action == "train")) / n_agents)
        mean_reward.append(float(np.mean(hx.agent_total_rewards)) if hx.agent_total_rewards else 0.0)

    return {
        "winning_bid_ratio": winning_bid_ratio,
        "training_rate": training_rate,
        "mean_reward": mean_reward,
    }


def _agg_by_round(series_list: List[List[float]]) -> Dict[int, List[float]]:
    per_round: Dict[int, List[float]] = defaultdict(list)
    for series in series_list:
        for i, value in enumerate(series):
            per_round[i].append(value)
    return per_round


def _plot_trace(ax, per_round: Dict[int, List[float]], title: str, ylabel: str) -> None:
    if not per_round:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Round")
        return

    rounds = sorted(per_round.keys())
    med = [float(np.median(per_round[r])) for r in rounds]
    p25 = [float(np.percentile(per_round[r], 25)) for r in rounds]
    p75 = [float(np.percentile(per_round[r], 75)) for r in rounds]

    ax.plot(rounds, med, linewidth=2)
    ax.fill_between(rounds, p25, p75, alpha=0.2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Round")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot standardized trace charts for selected runs.")
    parser.add_argument("--root", default="logs")
    parser.add_argument("--study", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--glob", dest="glob_pat", default="*.log")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    paths = []
    for path in discover_logs(root=args.root, study=args.study, variant=args.variant):
        if args.glob_pat and not fnmatch.fnmatch(path.name, args.glob_pat):
            continue
        if args.tag and args.tag not in path.as_posix():
            continue
        paths.append(path)

    if not paths:
        print("No matching logs found.")
        return 0

    collected = {"winning_bid_ratio": [], "training_rate": [], "mean_reward": []}
    for path in paths:
        try:
            exp_log = ExperimentLog.load(str(path))
        except Exception:
            continue
        traces = _collect_traces(exp_log)
        for key in collected:
            collected[key].append(traces[key])

    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.study and args.variant:
        out_dir = Path(args.root) / args.study / args.variant / "analysis"
    else:
        out_dir = Path(args.root) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_trace(axes[0], _agg_by_round(collected["winning_bid_ratio"]), "Winning Bid Ratio", "bid/base")
    _plot_trace(axes[1], _agg_by_round(collected["training_rate"]), "Training Rate", "fraction")
    _plot_trace(axes[2], _agg_by_round(collected["mean_reward"]), "Reward Trajectory", "mean reward")
    fig.tight_layout()

    out_path = out_dir / "trace_plots.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote trace plots -> {out_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
