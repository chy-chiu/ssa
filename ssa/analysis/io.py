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
    "scoring_rho",
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


def _is_policy_agent_type(type_name: object) -> bool:
    return "policyagent" in str(type_name).lower()


def _agent_ids_from_spec(spec: dict) -> List[str]:
    ids = spec.get("ids")
    if isinstance(ids, list):
        return [str(agent_id) for agent_id in ids]

    count = int(spec.get("count", 0) or 0)
    template = spec.get("id_template")
    if template and count > 0:
        out = []
        for i in range(count):
            try:
                out.append(str(template).format(i=i))
            except Exception:
                continue
        return out
    return []


def _policy_agent_ids_from_config(exp_log: ExperimentLog) -> set[str]:
    config = dict(exp_log.config or {})
    runner_cfg = config.get("runner_config") or {}
    specs = runner_cfg.get("agents") or []
    policy_ids: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if not _is_policy_agent_type(spec.get("type")):
            continue
        policy_ids.update(_agent_ids_from_spec(spec))
    return policy_ids


def _policy_agent_ids_fallback(agent_ids: List[str]) -> set[str]:
    policy_ids: set[str] = set()
    for agent_id in agent_ids:
        tag = str(agent_id).upper()
        if tag.startswith("POL-") or tag.startswith("PL-") or tag.startswith("FIXPL"):
            policy_ids.add(str(agent_id))
    return policy_ids


def get_non_policy_agent_ids(exp_log: ExperimentLog) -> List[str]:
    policy_ids = _policy_agent_ids_from_config(exp_log)
    if not policy_ids:
        policy_ids = _policy_agent_ids_fallback(exp_log.agent_ids)
    return [agent_id for agent_id in exp_log.agent_ids if agent_id not in policy_ids]


def _run_metrics(exp_log: ExperimentLog) -> dict:
    selected_agent_ids = get_non_policy_agent_ids(exp_log)
    selected_agent_set = set(selected_agent_ids)
    selected_agent_indices = {i for i, agent_id in enumerate(exp_log.agent_ids) if agent_id in selected_agent_set}

    n_rounds = len(exp_log.history)
    n_agents = len(selected_agent_ids)
    denom = max(n_rounds * max(n_agents, 1), 1)

    train_actions = 0
    winning_bid_ratios = []
    all_bid_ratios = []
    for hx in exp_log.history:
        for agent_idx, action in enumerate(hx.agent_actions):
            if agent_idx in selected_agent_indices and action.action == "train":
                train_actions += 1
        for job_id, bids in hx.agent_bids.items():
            base_price = hx.base_prices.get(job_id, 0)
            if base_price <= 0:
                continue
            for agent_idx, bid_price in bids.items():
                try:
                    idx = int(agent_idx)
                except Exception:
                    continue
                if idx in selected_agent_indices:
                    all_bid_ratios.append(float(bid_price) / float(base_price))
        for job_id, winning_price in hx.winning_prices.items():
            winner_idx = hx.matched_jobs.get(job_id, -1)
            if winner_idx not in selected_agent_indices:
                continue
            base_price = hx.base_prices.get(job_id, 0)
            if base_price > 0:
                winning_bid_ratios.append(float(winning_price) / float(base_price))

    train_rate = train_actions / denom
    train_rate_10 = 1.0 - (1.0 - min(max(train_rate, 0.0), 1.0)) ** 10
    mean_bid_ratio = float(pd.Series(all_bid_ratios).mean()) if all_bid_ratios else 0.0
    mean_winning_bid_ratio = float(pd.Series(winning_bid_ratios).mean()) if winning_bid_ratios else 0.0
    realized_job_performances = [
        float(perf.performance)
        for perf in (exp_log.job_performance or [])
        if int(getattr(perf, "round", -1)) >= 0 and str(perf.agent_id) in selected_agent_set
    ]
    mean_job_performance = (
        float(pd.Series(realized_job_performances).mean()) if realized_job_performances else 0.0
    )

    final_rewards = (
        pd.Series(exp_log.agent_total_rewards[-1], index=exp_log.agent_ids) if exp_log.history else pd.Series(dtype=float)
    )
    if not final_rewards.empty:
        final_rewards = final_rewards[final_rewards.index.isin(selected_agent_ids)]
    ssa_rewards = final_rewards[final_rewards.index.str.upper().str.startswith("SSA")]
    control_rewards = final_rewards[~final_rewards.index.str.upper().str.startswith("SSA")]
    ssa_mean_reward = float(ssa_rewards.mean()) if not ssa_rewards.empty else 0.0
    control_mean_reward = float(control_rewards.mean()) if not control_rewards.empty else 0.0

    return {
        "n_rounds": n_rounds,
        "n_agents": n_agents,
        "n_jobs": len(exp_log.job_ids),
        "train_rate": train_rate,
        "train_rate_10": train_rate_10,
        "mean_bid_ratio": mean_bid_ratio,
        "mean_winning_bid_ratio": mean_winning_bid_ratio,
        "mean_job_performance": mean_job_performance,
        "ssa_mean_reward": ssa_mean_reward,
        "control_mean_reward": control_mean_reward,
        "ssa_minus_control_reward": ssa_mean_reward - control_mean_reward,
    }


def _run_index_row(path: Path, exp_log: ExperimentLog) -> dict:
    config = dict(exp_log.config or {})
    defaults = _path_defaults(path)
    metadata = {key: config.get(key) for key in REQUIRED_RUN_METADATA}
    total_tokens = _safe_float((exp_log.token_usage or {}).get("total_token_usage", {}).get("total_tokens", 0))
    completion_tokens = _safe_float(
        (exp_log.token_usage or {}).get("total_token_usage", {}).get("completion_tokens", 0)
    )
    prompt_tokens = _safe_float((exp_log.token_usage or {}).get("total_token_usage", {}).get("prompt_tokens", 0))

    return {
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


def append_run_index_row(
    log_path: str | Path,
    *,
    root: str | Path = "logs",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(log_path)
    exp_log = ExperimentLog.load(str(path))
    row = _run_index_row(path, exp_log)
    row_df = pd.DataFrame([row])

    out = Path(output_path) if output_path else Path(root) / "run_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        try:
            df = pd.read_csv(out)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if "path" in df.columns:
        df = df[df["path"] != row["path"]]

    all_cols = list(dict.fromkeys([*df.columns.tolist(), *row_df.columns.tolist()]))
    df = df.reindex(columns=all_cols)
    row_df = row_df.reindex(columns=all_cols)
    merged = row_df if df.empty else pd.concat([df, row_df], ignore_index=True)
    sort_keys = [key for key in ("study", "variant", "run_name", "replicate_id") if key in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys).reset_index(drop=True)
    merged.to_csv(out, index=False)
    return merged


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

        rows.append(_run_index_row(path, exp_log))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["study", "variant", "run_name", "replicate_id"]).reset_index(drop=True)

    out = Path(output_path) if output_path else Path(root) / "run_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
