from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from ssa.common import ExperimentLog

REQUIRED_RUN_METADATA = (
    "study",
    "variant",
    "reviewer_target",
    "hypothesis_id",
    "git_commit",
    "effective_seed",
    "scoring_mode",
    "rep_update_mode",
    "agent_mix",
)

_PATH_LAYOUT_RE = re.compile(r"logs/(?P<study>[^/]+)/(?P<variant>[^/]+)/(?P<base>[^/]+)\.log$")
_RUN_RE = re.compile(r"(?P<run_name>.+)_(?P<replicate_id>\d+)$")


def discover_logs(
    root: str | Path = "logs",
    *,
    study: Optional[str] = None,
    variant: Optional[str] = None,
) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    if study and variant:
        search_root = root_path / study / variant
    elif study:
        search_root = root_path / study
    else:
        search_root = root_path

    if not search_root.exists():
        return []

    return sorted(path for path in search_root.rglob("*.log") if path.is_file())


def load_experiment_logs(paths: Iterable[str | Path]) -> List[ExperimentLog]:
    logs: List[ExperimentLog] = []
    for path in paths:
        try:
            logs.append(ExperimentLog.load(str(path)))
        except Exception:
            continue
    return logs


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _path_defaults(path: Path) -> dict:
    raw = path.as_posix()
    match = _PATH_LAYOUT_RE.search(raw)
    if not match:
        return {"study": "unknown", "variant": "unknown", "run_name": path.stem, "replicate_id": -1}

    study = match.group("study")
    variant = match.group("variant")
    base = match.group("base")
    run_match = _RUN_RE.match(base)
    if run_match:
        run_name = run_match.group("run_name")
        replicate_id = int(run_match.group("replicate_id"))
    else:
        run_name = base
        replicate_id = -1
    return {"study": study, "variant": variant, "run_name": run_name, "replicate_id": replicate_id}


def _run_metrics(exp_log: ExperimentLog) -> dict:
    n_rounds = len(exp_log.history)
    n_agents = len(exp_log.agent_ids)
    denom = max(n_rounds * max(n_agents, 1), 1)

    train_actions = 0
    winning_bid_ratios = []
    for hx in exp_log.history:
        for action in hx.agent_actions:
            if action.action == "train":
                train_actions += 1
        for job_id, winning_price in hx.winning_prices.items():
            base_price = hx.base_prices.get(job_id, 0)
            if base_price > 0:
                winning_bid_ratios.append(float(winning_price) / float(base_price))

    train_rate = train_actions / denom
    mean_winning_bid_ratio = float(pd.Series(winning_bid_ratios).mean()) if winning_bid_ratios else 0.0

    final_rewards = pd.Series(exp_log.agent_total_rewards[-1], index=exp_log.agent_ids) if exp_log.history else pd.Series(dtype=float)
    ssa_rewards = final_rewards[final_rewards.index.str.upper().str.startswith("SSA")]
    control_rewards = final_rewards[~final_rewards.index.str.upper().str.startswith("SSA")]
    ssa_mean_reward = float(ssa_rewards.mean()) if not ssa_rewards.empty else 0.0
    control_mean_reward = float(control_rewards.mean()) if not control_rewards.empty else 0.0

    return {
        "n_rounds": n_rounds,
        "n_agents": n_agents,
        "n_jobs": len(exp_log.job_ids),
        "train_rate": train_rate,
        "mean_winning_bid_ratio": mean_winning_bid_ratio,
        "ssa_mean_reward": ssa_mean_reward,
        "control_mean_reward": control_mean_reward,
        "ssa_minus_control_reward": ssa_mean_reward - control_mean_reward,
    }


def build_run_index(
    root: str | Path = "logs",
    *,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    for path in discover_logs(root):
        try:
            exp_log = ExperimentLog.load(str(path))
        except Exception:
            continue

        config = dict(exp_log.config or {})
        defaults = _path_defaults(path)
        metadata = {key: config.get(key) for key in REQUIRED_RUN_METADATA}
        total_tokens = _safe_float((exp_log.token_usage or {}).get("total_token_usage", {}).get("total_tokens", 0))
        completion_tokens = _safe_float(
            (exp_log.token_usage or {}).get("total_token_usage", {}).get("completion_tokens", 0)
        )
        prompt_tokens = _safe_float((exp_log.token_usage or {}).get("total_token_usage", {}).get("prompt_tokens", 0))

        row = {
            "path": path.as_posix(),
            "study": config.get("study", defaults["study"]),
            "variant": config.get("variant", defaults["variant"]),
            "run_name": config.get("run_name", defaults["run_name"]),
            "replicate_id": int(config.get("replicate_id", defaults["replicate_id"])),
            "suite_seed": config.get("suite_seed"),
            "effective_seed": config.get("effective_seed"),
            "open_bidding": bool(config.get("open_bidding", False)),
            "performance_pay": bool(config.get("performance_pay", True)),
            "total_tokens": total_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            **metadata,
            **_run_metrics(exp_log),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["study", "variant", "run_name", "replicate_id"]).reset_index(drop=True)

    out = Path(output_path) if output_path else Path(root) / "run_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
