from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy, rankdata

from ssa.common import ExperimentLog


def compute_trace_means(traces: Iterable[Sequence[Tuple[float, float]]], max_value: float = 2.0) -> np.ndarray:
    time_values = defaultdict(list)
    for trace in traces:
        for point in trace:
            if len(point) < 2:
                continue
            t, v = point[0], point[1]
            if np.isfinite(t) and np.isfinite(v) and v <= max_value:
                time_values[float(t)].append(float(v))

    if not time_values:
        return np.empty((0, 2), dtype=float)

    return np.array([(t, float(np.mean(vs))) for t, vs in sorted(time_values.items())], dtype=float)


def interp_trace(trace: Sequence[Tuple[float, float]] | np.ndarray, n_steps: int = 100) -> np.ndarray:
    if n_steps <= 0:
        return np.array([], dtype=float)
    if trace is None:
        return np.zeros(n_steps, dtype=float)
    if isinstance(trace, np.ndarray):
        arr = trace
    else:
        arr = np.array(list(trace), dtype=float) if trace else np.empty((0, 2), dtype=float)
    if arr.size == 0:
        return np.zeros(n_steps, dtype=float)

    times = arr[:, 0]
    values = arr[:, 1]
    if len(times) == 1:
        return np.full(n_steps, float(values[0]), dtype=float)

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    x = np.arange(n_steps, dtype=float)
    return np.interp(x, times, values)


def recovery_score(reward_traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if reward_traces.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if reward_traces.ndim != 2:
        raise ValueError("reward_traces must be 2D (n_agents, n_steps)")

    n_agents, n_steps = reward_traces.shape
    if n_steps <= 1:
        return np.zeros(n_agents, dtype=float), np.zeros(n_agents, dtype=float)

    ranks = np.array([rankdata(-reward_traces[:, t]) for t in range(n_steps)]).T
    improvements = np.diff(ranks, axis=1) < 0
    recovery_rates = improvements.sum(axis=1) / float(n_steps - 1)
    rank_spread = ranks.max(axis=1) - ranks.min(axis=1)
    return recovery_rates, rank_spread


def gini(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    if np.min(arr) < 0:
        arr = arr - np.min(arr)
    total = np.sum(arr)
    if total <= 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    index = np.arange(1, n + 1, dtype=float)
    return float((np.sum((2 * index - n - 1) * sorted_arr)) / (n * total))


def _calculate_specialization(skill_vector: Sequence[float], init: float = 40.0) -> float:
    skills = np.array(skill_vector, dtype=float) - init
    skills = np.clip(skills, a_min=0.0, a_max=None)
    if skills.size == 0:
        return 0.0
    total = float(np.sum(skills))
    if total <= 0:
        return 0.0
    probabilities = skills / total
    h = float(entropy(probabilities))
    h_max = float(np.log(len(skills))) if len(skills) > 1 else 1.0
    if h_max <= 0:
        return 0.0
    return float(1 - h / h_max)


def _final_skill_values(skill_history: dict) -> List[float]:
    values = []
    for _, history in (skill_history or {}).items():
        if isinstance(history, (tuple, list)) and len(history) >= 2 and isinstance(history[1], list) and history[1]:
            values.append(float(history[1][-1]))
        elif isinstance(history, (int, float)):
            values.append(float(history))
    return values


def get_summary_df(exp_log: ExperimentLog, fp: str = "") -> pd.DataFrame:
    rows = []
    for step, hx in enumerate(exp_log.history):
        for agent_idx, action in enumerate(hx.agent_actions):
            agent_id = exp_log.agent_ids[agent_idx]
            if action.action == "train":
                train_target = action.targets[0][0] if action.targets else None
                rows.append({"step": step, "agent_id": agent_id, "action": "train", "train_target": train_target})
                continue

            if action.action == "error":
                rows.append({"step": step, "agent_id": agent_id, "action": "error"})
                continue

            all_targets = [target[0] for target in action.targets]
            winning_priority = []
            for priority, job_id in enumerate(all_targets[:5]):
                if hx.matched_jobs.get(job_id, -1) == agent_idx:
                    winning_priority.append(priority + 1)

            if not all_targets:
                rows.append({"step": step, "agent_id": agent_id, "action": "bid", "winrate": 0.0})
                continue

            base_prices = [hx.base_prices[t] for t in all_targets if t in hx.base_prices]
            rows.append(
                {
                    "step": step,
                    "agent_id": agent_id,
                    "action": "bid",
                    "winrate": float(len(winning_priority) / max(min(len(all_targets), 3), 1)),
                    "winning_priority": float(np.mean(winning_priority)) if winning_priority else 0.0,
                    "avg_base_price": float(np.mean(base_prices)) if base_prices else 0.0,
                    "top_base_price": float(hx.base_prices.get(all_targets[0], 0.0)),
                }
            )

    action_df = pd.DataFrame(rows)
    if action_df.empty:
        action_df = pd.DataFrame({"agent_id": exp_log.agent_ids})
    if "action" not in action_df.columns:
        action_df["action"] = "error"

    action_df = action_df.join(pd.get_dummies(action_df["action"]))
    for col in ("bid", "train", "error"):
        if col not in action_df.columns:
            action_df[col] = 0.0
    if "train_target" not in action_df.columns:
        action_df["train_target"] = None
    if "winrate" not in action_df.columns:
        action_df["winrate"] = 0.0
    if "winning_priority" not in action_df.columns:
        action_df["winning_priority"] = 0.0
    if "avg_base_price" not in action_df.columns:
        action_df["avg_base_price"] = 0.0
    if "top_base_price" not in action_df.columns:
        action_df["top_base_price"] = 0.0

    rewards_df = pd.DataFrame(
        [
            {"run": fp, "agent_id": agent.id, "reward": float(agent.total_reward), "atype": str(agent.id).split("-")[0]}
            for agent in exp_log.agents
        ]
    ).set_index("agent_id")
    if rewards_df.empty:
        rewards_df = pd.DataFrame({"run": fp, "reward": 0.0, "atype": ""}, index=exp_log.agent_ids)

    rewards_df["rank"] = rewards_df["reward"].rank(ascending=False, method="average")
    total_reward = float(rewards_df["reward"].sum())
    rewards_df["reward_normalized"] = rewards_df["reward"] / total_reward if total_reward else 0.0

    summary_df = action_df.groupby("agent_id").agg(
        {
            "winrate": "mean",
            "top_base_price": "mean",
            "avg_base_price": "mean",
            "bid": "mean",
            "train": "mean",
            "error": "mean",
            "train_target": "nunique",
        }
    )

    all_winning_bids = []
    all_agent_bids = []
    for hx in exp_log.history:
        agent_bids = defaultdict(list)
        agent_winning_bids = defaultdict(list)
        for job_id, bids in hx.agent_bids_normalized.items():
            for agent_idx, norm_bid in bids.items():
                agent_id = exp_log.agent_ids[int(agent_idx)]
                agent_bids[agent_id].append(float(norm_bid))
                if hx.matched_jobs.get(job_id, -1) == int(agent_idx):
                    agent_winning_bids[agent_id].append(float(norm_bid))
        all_winning_bids.append({k: float(np.mean(v)) for k, v in agent_winning_bids.items() if v})
        all_agent_bids.append({k: float(np.mean(v)) for k, v in agent_bids.items() if v})

    summary_df["all_bids"] = pd.DataFrame(all_agent_bids).mean()
    summary_df["winning_bids"] = pd.DataFrame(all_winning_bids).mean()
    summary_df["win_prio"] = action_df.query("winrate > 0").groupby("agent_id")["winning_priority"].mean()
    job_perf_rows = [
        {"agent_id": perf.agent_id, "performance": float(perf.performance)}
        for perf in (exp_log.job_performance or [])
        if int(getattr(perf, "round", -1)) >= 0
    ]
    if job_perf_rows:
        mean_perf_by_agent = pd.DataFrame(job_perf_rows).groupby("agent_id")["performance"].mean()
        summary_df["mean_job_performance"] = mean_perf_by_agent
    else:
        summary_df["mean_job_performance"] = 0.0

    token_rows = (exp_log.token_usage or {}).get("agent_token_usage", [])
    completion_tokens = {}
    total_tokens = {}
    for agent_id, token_row in zip(exp_log.agent_ids, token_rows):
        totals = token_row.get("total_token_usage", {})
        completion_tokens[agent_id] = totals.get("completion_tokens", 0)
        total_tokens[agent_id] = totals.get("total_tokens", 0)
    summary_df["completion_tokens"] = pd.Series(completion_tokens)
    summary_df["total_tokens"] = pd.Series(total_tokens)

    reward_traces = np.array(exp_log.agent_total_rewards, dtype=float).T if exp_log.history else np.empty((0, 0))
    scores, rank_jump = recovery_score(reward_traces) if reward_traces.size else (np.array([]), np.array([]))
    summary_df["recovery"] = pd.Series({aid: val for aid, val in zip(exp_log.agent_ids, scores)})
    summary_df["rank_jump"] = pd.Series({aid: val for aid, val in zip(exp_log.agent_ids, rank_jump)})

    denom = summary_df["train"] + summary_df["bid"]
    summary_df["train_p"] = np.where(denom > 0, summary_df["train"] / denom, 0.0)

    skill_rep_rows = []
    for agent in exp_log.agents:
        final_skills = _final_skill_values(agent.skill_history)
        reps = [float(r[1]) for r in (agent.reputation or {}).values() if isinstance(r, (tuple, list)) and len(r) >= 2]
        skill_rep_rows.append(
            {
                "agent_id": agent.id,
                "skill_sum": float(np.sum(final_skills)) if final_skills else 0.0,
                "skill_max": float(np.max(final_skills)) if final_skills else 0.0,
                "skill_spec": _calculate_specialization(final_skills, init=0.0),
                "rep_avg": float(np.mean(reps)) if reps else 0.0,
                "rep_max": float(np.max(reps)) if reps else 0.0,
                "rep_spec": _calculate_specialization(reps, init=0.0),
            }
        )

    skill_rep_df = pd.DataFrame(skill_rep_rows).set_index("agent_id") if skill_rep_rows else pd.DataFrame()
    out = rewards_df.join(summary_df).join(skill_rep_df).reset_index()

    expected = [
        "run",
        "agent_id",
        "atype",
        "reward",
        "reward_normalized",
        "rank",
        "winrate",
        "win_prio",
        "recovery",
        "rank_jump",
        "top_base_price",
        "avg_base_price",
        "all_bids",
        "winning_bids",
        "mean_job_performance",
        "train_p",
        "train_target",
        "skill_sum",
        "skill_max",
        "skill_spec",
        "rep_avg",
        "rep_max",
        "rep_spec",
        "total_tokens",
        "completion_tokens",
    ]
    for col in expected:
        if col not in out.columns:
            out[col] = 0.0 if col not in {"run", "agent_id", "atype"} else ""
    return out[expected]
