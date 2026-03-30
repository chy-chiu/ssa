from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from ssa.common import (
    AgentActionResponse,
    AgentHistory,
    AgentLog,
    AgentPerformance,
    ExperimentLog,
    JobHistory,
    RoundData,
    SubAgentLog,
)

_VALID_ACTIONS = {"bid", "train", "error"}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _normalize_targets(targets: Any, action: str = "error") -> List[Tuple[Any, Any]]:
    out: List[Tuple[Any, Any]] = []
    if not isinstance(targets, list):
        return out

    for target in targets:
        if isinstance(target, (list, tuple)) and target:
            item = target[0]
            if len(target) > 1:
                price = target[1]
            else:
                price = -1 if action == "train" else 0
            out.append((item, price))
        elif isinstance(target, str):
            out.append((target, -1 if action == "train" else 0))
    return out


def _action_to_dict(raw_action: Any, strip_reasoning: bool) -> Dict[str, Any]:
    if isinstance(raw_action, AgentActionResponse):
        raw = raw_action.model_dump()
    else:
        raw = _ensure_dict(raw_action)

    action = str(raw.get("action", "error") or "error").lower()
    if action not in _VALID_ACTIONS:
        action = "error"

    return {
        "reasoning": "" if strip_reasoning else str(raw.get("reasoning", "") or ""),
        "action": action,
        "targets": _normalize_targets(raw.get("targets", []), action=action),
    }


def _coerce_agent_value_map(
    raw_map: Any, *, allow_none: bool = False, value_default: float = 0.0
) -> Dict[str, Dict[int, float | None]]:
    out: Dict[str, Dict[int, float | None]] = {}
    for job_id, value_map in _ensure_dict(raw_map).items():
        converted: Dict[int, float | None] = {}
        for agent_idx, raw_value in _ensure_dict(value_map).items():
            idx = _to_int(agent_idx, default=-1)
            if idx < 0:
                continue
            if raw_value is None and allow_none:
                converted[idx] = None
            else:
                converted[idx] = _to_float(raw_value, default=value_default)
        out[str(job_id)] = converted
    return out


def _coerce_job_performance_map(raw_map: Any) -> Dict[str, Tuple[int, float]]:
    out: Dict[str, Tuple[int, float]] = {}
    for job_id, pair in _ensure_dict(raw_map).items():
        idx = -1
        perf = 0.0
        if isinstance(pair, (list, tuple)):
            if len(pair) > 0:
                idx = _to_int(pair[0], default=-1)
            if len(pair) > 1:
                perf = _to_float(pair[1], default=0.0)
        out[str(job_id)] = (idx, perf)
    return out


def _coerce_skill_rows(rows: Any) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    for row in _ensure_list(rows):
        row_dict = _ensure_dict(row)
        out.append({str(task_id): _to_int(level, default=0) for task_id, level in row_dict.items()})
    return out


def build_lean_payload(
    raw: Mapping[str, Any],
    *,
    strip_reasoning: bool = True,
    drop_agent_trace: bool = True,
    drop_subagent_traces: bool = True,
    drop_subagents: bool = True,
    drop_market_history: bool = True,
    drop_agent_history_str: bool = True,
    drop_job_histories: bool = True,
) -> Dict[str, Any]:
    payload = dict(raw)
    history_rows = _ensure_list(payload.get("history"))
    agent_rows = _ensure_list(payload.get("agents"))

    lean_history = []
    for hx in history_rows:
        hx_dict = _ensure_dict(hx)
        lean_history.append(
            {
                "round": _to_int(hx_dict.get("round"), default=0),
                "base_prices": _ensure_dict(hx_dict.get("base_prices")),
                "agent_actions": [
                    _action_to_dict(action, strip_reasoning=strip_reasoning)
                    for action in _ensure_list(hx_dict.get("agent_actions"))
                ],
                "agent_bids": _ensure_dict(hx_dict.get("agent_bids")),
                "agent_bids_normalized": _ensure_dict(hx_dict.get("agent_bids_normalized")),
                "agent_preferences": _ensure_list(hx_dict.get("agent_preferences")),
                "winning_prices": _ensure_dict(hx_dict.get("winning_prices")),
                "unranked_agent_scores": _ensure_dict(hx_dict.get("unranked_agent_scores")),
                "reranked_agent_scores": _ensure_dict(hx_dict.get("reranked_agent_scores")),
                "market_preference": _ensure_dict(hx_dict.get("market_preference")),
                "matched_jobs": _ensure_dict(hx_dict.get("matched_jobs")),
                "unmatched_agents": _ensure_list(hx_dict.get("unmatched_agents")),
                "unmatched_jobs": _ensure_list(hx_dict.get("unmatched_jobs")),
                "prev_reputation": _ensure_dict(hx_dict.get("prev_reputation")),
                "agent_reputation": _ensure_dict(hx_dict.get("agent_reputation")),
                "agent_skills": _ensure_list(hx_dict.get("agent_skills")),
                "job_performance": _ensure_dict(hx_dict.get("job_performance")),
                "agent_round_rewards": _ensure_list(hx_dict.get("agent_round_rewards")),
                "agent_total_rewards": _ensure_list(hx_dict.get("agent_total_rewards")),
            }
        )

    lean_agents = []
    for agent in agent_rows:
        agent_dict = _ensure_dict(agent)
        lean_history_rows = []
        for entry in _ensure_list(agent_dict.get("agent_history")):
            entry_dict = _ensure_dict(entry)
            lean_history_rows.append(
                {
                    "round": _to_int(entry_dict.get("round"), default=0),
                    "listings": _ensure_dict(entry_dict.get("listings")),
                    "agent_action": _action_to_dict(
                        entry_dict.get("agent_action"), strip_reasoning=strip_reasoning
                    ),
                    "allocated_jobs": [] if drop_job_histories else _ensure_list(entry_dict.get("allocated_jobs")),
                    "unallocated_jobs": [] if drop_job_histories else _ensure_list(entry_dict.get("unallocated_jobs")),
                    "total_reward": _to_float(entry_dict.get("total_reward"), default=0.0),
                    "reputation_update": _ensure_dict(entry_dict.get("reputation_update")),
                    "training_performed": str(entry_dict.get("training_performed", "") or ""),
                }
            )

        if drop_subagents:
            lean_subagents: Dict[str, Any] = {}
        else:
            lean_subagents = {}
            for task_id, subagent in _ensure_dict(agent_dict.get("subagents")).items():
                subagent_dict = _ensure_dict(subagent)
                lean_subagents[str(task_id)] = {
                    "knowledge_base": _ensure_dict(subagent_dict.get("knowledge_base")),
                    "token_usage": _ensure_dict(subagent_dict.get("token_usage")),
                    "trace": [] if drop_subagent_traces else _ensure_list(subagent_dict.get("trace")),
                }

        lean_agents.append(
            {
                "id": str(agent_dict.get("id", "")),
                "idx": _to_int(agent_dict.get("idx"), default=-1),
                "agent_history": lean_history_rows,
                "agent_history_str": []
                if drop_agent_history_str
                else [str(x) for x in _ensure_list(agent_dict.get("agent_history_str"))],
                "market_history": [] if drop_market_history else _ensure_list(agent_dict.get("market_history")),
                "skill_history": _ensure_dict(agent_dict.get("skill_history")),
                "reputation": _ensure_dict(agent_dict.get("reputation")),
                "total_reward": _to_float(agent_dict.get("total_reward"), default=0.0),
                "trace": [] if drop_agent_trace else _ensure_list(agent_dict.get("trace")),
                "token_usage": _ensure_dict(agent_dict.get("token_usage")),
                "subagents": lean_subagents,
            }
        )

    return {
        "config": _ensure_dict(payload.get("config")),
        "jobs": _ensure_list(payload.get("jobs")),
        "agent_ids": [str(agent_id) for agent_id in _ensure_list(payload.get("agent_ids"))],
        "task_ids": [str(task_id) for task_id in _ensure_list(payload.get("task_ids"))],
        "job_ids": [str(job_id) for job_id in _ensure_list(payload.get("job_ids"))],
        "job_to_task_id": _ensure_dict(payload.get("job_to_task_id")),
        "history": lean_history,
        "job_performance": _ensure_list(payload.get("job_performance")),
        "agents": lean_agents,
        "token_usage": _ensure_dict(payload.get("token_usage")),
    }


def _construct_action(raw_action: Any) -> AgentActionResponse:
    action_dict = _action_to_dict(raw_action, strip_reasoning=False)
    return AgentActionResponse.model_construct(
        reasoning=action_dict["reasoning"],
        action=action_dict["action"],
        targets=action_dict["targets"],
    )


def _construct_job_history(raw_job_history: Any) -> JobHistory:
    row = _ensure_dict(raw_job_history)
    return JobHistory.model_construct(
        job_id=str(row.get("job_id", "")),
        task_id=str(row.get("task_id", "")),
        base_price=_to_float(row.get("base_price"), default=0.0),
        bid_price=_to_float(row.get("bid_price"), default=0.0),
        performance=_to_float(row.get("performance"), default=0.0),
        adjusted_reward=_to_float(row.get("adjusted_reward"), default=0.0),
        old_reputation=_to_float(row.get("old_reputation"), default=0.0),
        new_reputation=_to_float(row.get("new_reputation"), default=0.0),
    )


def _construct_agent_history(raw_agent_history: Any) -> AgentHistory:
    row = _ensure_dict(raw_agent_history)
    return AgentHistory.model_construct(
        round=_to_int(row.get("round"), default=0),
        listings=_ensure_dict(row.get("listings")),
        agent_action=_construct_action(row.get("agent_action")),
        allocated_jobs=[_construct_job_history(v) for v in _ensure_list(row.get("allocated_jobs"))],
        unallocated_jobs=[_construct_job_history(v) for v in _ensure_list(row.get("unallocated_jobs"))],
        total_reward=_to_float(row.get("total_reward"), default=0.0),
        reputation_update=_ensure_dict(row.get("reputation_update")),
        training_performed=str(row.get("training_performed", "") or ""),
    )


def _construct_subagent_log(raw_subagent_log: Any) -> SubAgentLog:
    row = _ensure_dict(raw_subagent_log)
    trace = []
    for entry in _ensure_list(row.get("trace")):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            trace.append((str(entry[0]), str(entry[1])))
    return SubAgentLog.model_construct(
        knowledge_base={str(k): str(v) for k, v in _ensure_dict(row.get("knowledge_base")).items()},
        token_usage=_ensure_dict(row.get("token_usage")),
        trace=trace,
    )


def _construct_agent_log(raw_agent_log: Any) -> AgentLog:
    row = _ensure_dict(raw_agent_log)
    return AgentLog.model_construct(
        id=str(row.get("id", "")),
        idx=_to_int(row.get("idx"), default=-1),
        agent_history=[_construct_agent_history(v) for v in _ensure_list(row.get("agent_history"))],
        agent_history_str=[str(x) for x in _ensure_list(row.get("agent_history_str"))],
        market_history=_ensure_list(row.get("market_history")),
        skill_history=_ensure_dict(row.get("skill_history")),
        reputation=_ensure_dict(row.get("reputation")),
        total_reward=_to_float(row.get("total_reward"), default=0.0),
        trace=_ensure_list(row.get("trace")),
        token_usage=_ensure_dict(row.get("token_usage")),
        subagents={str(task_id): _construct_subagent_log(v) for task_id, v in _ensure_dict(row.get("subagents")).items()},
    )


def _construct_round_data(raw_round: Any) -> RoundData:
    row = _ensure_dict(raw_round)
    prev_reputation = {
        str(task_id): [_to_float(x, default=0.0) for x in _ensure_list(values)]
        for task_id, values in _ensure_dict(row.get("prev_reputation")).items()
    }
    agent_reputation = {
        str(task_id): [_to_float(x, default=0.0) for x in _ensure_list(values)]
        for task_id, values in _ensure_dict(row.get("agent_reputation")).items()
    }
    matched_jobs = {str(job_id): _to_int(agent_idx, default=-1) for job_id, agent_idx in _ensure_dict(row.get("matched_jobs")).items()}

    return RoundData.model_construct(
        round=_to_int(row.get("round"), default=0),
        base_prices={str(job_id): _to_float(price, default=0.0) for job_id, price in _ensure_dict(row.get("base_prices")).items()},
        agent_actions=[_construct_action(v) for v in _ensure_list(row.get("agent_actions"))],
        agent_bids=_coerce_agent_value_map(row.get("agent_bids"), allow_none=False, value_default=0.0),
        agent_bids_normalized=_coerce_agent_value_map(
            row.get("agent_bids_normalized"), allow_none=False, value_default=0.0
        ),
        agent_preferences=[[str(job_id) for job_id in _ensure_list(pref)] for pref in _ensure_list(row.get("agent_preferences"))],
        winning_prices={str(job_id): _to_float(price, default=0.0) for job_id, price in _ensure_dict(row.get("winning_prices")).items()},
        unranked_agent_scores=_coerce_agent_value_map(row.get("unranked_agent_scores"), allow_none=True, value_default=0.0),
        reranked_agent_scores=_coerce_agent_value_map(row.get("reranked_agent_scores"), allow_none=True, value_default=0.0),
        market_preference={
            str(job_id): [_to_int(agent_idx, default=-1) for agent_idx in _ensure_list(agent_order)]
            for job_id, agent_order in _ensure_dict(row.get("market_preference")).items()
        },
        matched_jobs=matched_jobs,
        unmatched_agents=[_to_int(agent_idx, default=-1) for agent_idx in _ensure_list(row.get("unmatched_agents"))],
        unmatched_jobs=[str(job_id) for job_id in _ensure_list(row.get("unmatched_jobs"))],
        prev_reputation=prev_reputation,
        agent_reputation=agent_reputation,
        agent_skills=_coerce_skill_rows(row.get("agent_skills")),
        job_performance=_coerce_job_performance_map(row.get("job_performance")),
        agent_round_rewards=[_to_float(value, default=0.0) for value in _ensure_list(row.get("agent_round_rewards"))],
        agent_total_rewards=[_to_float(value, default=0.0) for value in _ensure_list(row.get("agent_total_rewards"))],
    )


def _construct_agent_performance(raw_agent_performance: Any) -> AgentPerformance:
    row = _ensure_dict(raw_agent_performance)
    return AgentPerformance.model_construct(
        agent_idx=_to_int(row.get("agent_idx"), default=-1),
        agent_id=str(row.get("agent_id", "")),
        round=_to_int(row.get("round"), default=-1),
        task_id=str(row.get("task_id", "")),
        job_id=str(row.get("job_id", "")),
        performance=_to_float(row.get("performance"), default=0.0),
        reputation=_to_float(row.get("reputation"), default=0.0),
    )


def _derive_job_ids(jobs: Iterable[Any]) -> List[str]:
    out = []
    for job in jobs:
        job_dict = _ensure_dict(job)
        job_id = str(job_dict.get("id", "")).strip()
        if job_id:
            out.append(job_id)
    return out


def _derive_task_ids(jobs: Iterable[Any]) -> List[str]:
    out = []
    seen = set()
    for job in jobs:
        job_dict = _ensure_dict(job)
        task_id = str(job_dict.get("task_id", "")).strip()
        if task_id and task_id not in seen:
            seen.add(task_id)
            out.append(task_id)
    return out


def _derive_job_to_task_id(jobs: Iterable[Any]) -> Dict[str, str]:
    out = {}
    for job in jobs:
        job_dict = _ensure_dict(job)
        job_id = str(job_dict.get("id", "")).strip()
        task_id = str(job_dict.get("task_id", "")).strip()
        if job_id:
            out[job_id] = task_id
    return out


def construct_experiment_log(payload: Mapping[str, Any]) -> ExperimentLog:
    payload_dict = dict(payload)
    jobs = _ensure_list(payload_dict.get("jobs"))
    agents_raw = _ensure_list(payload_dict.get("agents"))
    agent_logs = [_construct_agent_log(agent) for agent in agents_raw]

    agent_ids = [str(agent_id) for agent_id in _ensure_list(payload_dict.get("agent_ids"))]
    if not agent_ids:
        agent_ids = [agent.id for agent in agent_logs]

    job_ids = [str(job_id) for job_id in _ensure_list(payload_dict.get("job_ids"))]
    if not job_ids:
        job_ids = _derive_job_ids(jobs)

    task_ids = [str(task_id) for task_id in _ensure_list(payload_dict.get("task_ids"))]
    if not task_ids:
        task_ids = _derive_task_ids(jobs)

    job_to_task_id = _ensure_dict(payload_dict.get("job_to_task_id"))
    if not job_to_task_id:
        job_to_task_id = _derive_job_to_task_id(jobs)

    return ExperimentLog.model_construct(
        config=_ensure_dict(payload_dict.get("config")),
        jobs=jobs,
        agent_ids=agent_ids,
        task_ids=task_ids,
        job_ids=job_ids,
        job_to_task_id=job_to_task_id,
        history=[_construct_round_data(v) for v in _ensure_list(payload_dict.get("history"))],
        job_performance=[_construct_agent_performance(v) for v in _ensure_list(payload_dict.get("job_performance"))],
        agents=agent_logs,
        token_usage=_ensure_dict(payload_dict.get("token_usage")),
    )


def load_experiment_log_fast(
    filepath: str | Path,
    *,
    lean: bool = True,
    validate: bool = False,
    strip_reasoning: bool = True,
    drop_agent_trace: bool = True,
    drop_subagent_traces: bool = True,
    drop_subagents: bool = True,
    drop_market_history: bool = True,
    drop_agent_history_str: bool = True,
    drop_job_histories: bool = True,
) -> ExperimentLog:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    payload: Mapping[str, Any]
    if lean:
        payload = build_lean_payload(
            raw,
            strip_reasoning=strip_reasoning,
            drop_agent_trace=drop_agent_trace,
            drop_subagent_traces=drop_subagent_traces,
            drop_subagents=drop_subagents,
            drop_market_history=drop_market_history,
            drop_agent_history_str=drop_agent_history_str,
            drop_job_histories=drop_job_histories,
        )
    else:
        payload = raw

    if validate:
        return ExperimentLog.model_validate(payload)
    return construct_experiment_log(payload)


def write_lean_log(
    input_path: str | Path,
    output_path: str | Path,
    *,
    strip_reasoning: bool = True,
    drop_agent_trace: bool = True,
    drop_subagent_traces: bool = True,
    drop_subagents: bool = True,
    drop_market_history: bool = True,
    drop_agent_history_str: bool = True,
    drop_job_histories: bool = True,
) -> Path:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    lean_payload = build_lean_payload(
        raw,
        strip_reasoning=strip_reasoning,
        drop_agent_trace=drop_agent_trace,
        drop_subagent_traces=drop_subagent_traces,
        drop_subagents=drop_subagents,
        drop_market_history=drop_market_history,
        drop_agent_history_str=drop_agent_history_str,
        drop_job_histories=drop_job_histories,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lean_payload, f, separators=(",", ":"))
    return out_path
