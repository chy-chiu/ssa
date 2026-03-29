from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from ssa.common import ExperimentLog


def _parse_trace_entry(entry: Tuple) -> Tuple[str, str, str]:
    llm_reasoning = ""
    action = ""
    reasoning = ""

    if not isinstance(entry, tuple):
        return llm_reasoning, action, reasoning

    if len(entry) >= 3:
        llm_reasoning = str(entry[1] or "")
        payload = entry[2]
    elif len(entry) >= 2:
        payload = entry[1]
    else:
        payload = None

    if payload is not None:
        action = str(getattr(payload, "action", "") or "")
        reasoning = str(getattr(payload, "reasoning", "") or "")

    return llm_reasoning, action, reasoning


def format_trace(trace: Sequence[Tuple]) -> str:
    lines = []
    for idx, entry in enumerate(trace):
        _, action, reasoning = _parse_trace_entry(entry)
        lines.append(f"R{idx}: {action} | {reasoning}")
    return "\n".join(lines)


def format_trace_history(trace: Sequence[Tuple], history: Sequence) -> str:
    lines = []
    for idx, (entry, hx) in enumerate(zip(trace, history)):
        _, action, reasoning = _parse_trace_entry(entry)
        lines.append(f"R{idx}: reasoning={reasoning}")
        lines.append(f"R{idx}: action={action} | outcome={hx}")
        lines.append("=====")
    return "\n".join(lines)


def trace_to_dataframe(exp_log: ExperimentLog, run_id: str = "") -> pd.DataFrame:
    rows: List[Dict] = []
    for agent in exp_log.agents:
        for round_idx, entry in enumerate(agent.trace):
            llm_reasoning, action, reasoning = _parse_trace_entry(entry)
            targets = []
            if isinstance(entry, tuple) and len(entry) >= 3:
                payload = entry[2]
                raw_targets = getattr(payload, "targets", [])
                targets = list(raw_targets) if raw_targets else []
            round_reward = 0.0
            if round_idx < len(agent.agent_history):
                round_reward = float(agent.agent_history[round_idx].total_reward)
            rows.append(
                {
                    "run": run_id,
                    "agent_id": agent.id,
                    "round": round_idx,
                    "action": action,
                    "reasoning": reasoning,
                    "llm_reasoning": llm_reasoning,
                    "targets": targets,
                    "round_reward": round_reward,
                }
            )
    return pd.DataFrame(rows)


def batch_extract_traces(logs: Dict[str, ExperimentLog] | Iterable[Tuple[str, ExperimentLog]]) -> pd.DataFrame:
    if isinstance(logs, dict):
        items = logs.items()
    else:
        items = logs

    frames = []
    for run_id, exp_log in items:
        frames.append(trace_to_dataframe(exp_log, run_id=run_id))
    if not frames:
        return pd.DataFrame(columns=["run", "agent_id", "round", "action", "reasoning", "llm_reasoning", "targets"])
    return pd.concat(frames, ignore_index=True)
