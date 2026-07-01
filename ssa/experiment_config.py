from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dictionaries (override wins). Lists are replaced, not merged."""
    out: Dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            out[key] = deep_merge(base_value, override_value)
        else:
            out[key] = override_value
    return out


class ModelSpec(BaseModel):
    type: Literal[
        "none",
        "OpenAIClient",
        "OpenRouterClient",
        "init_openrouter_chat_model",
        "init_azure_model",
    ] = "none"
    params: Dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    type: Literal["ProxyTask"] = "ProxyTask"
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class JobSpec(BaseModel):
    id: str
    task_id: str
    base_reward: float
    job_p: float = 1.0
    noise: float = 0.0
    w_q: float = 0.6


class AgentSpec(BaseModel):
    type: str
    verbose: bool = False

    # Either explicit ids, or templated ids + count
    ids: Optional[List[str]] = None
    id_template: Optional[str] = None  # e.g. "SSA-{i}"
    count: int = 1

    model: Optional[ModelSpec] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    policy: Optional[Dict[str, Any]] = None  # for PolicyAgent.set_policy(...)


class MarketSpec(BaseModel):
    market_limit: int = 3
    market_pref_limit: int = 5
    agent_pref_limit: int = 5
    history_limit: int = 10
    skill_phi: float = 0.1
    rep_initial: float = 0.5
    rep_window: int = 5
    rep_sensitivity: float = 1.0
    rep_lambda: float = 0.5
    gumbel_t: float = 0.01
    performance_pay: bool = True
    open_bidding: bool = False


class JobUpdateEvent(BaseModel):
    at_step: int  # 0-based, applied before simulate_timestep()
    jobs: Dict[str, Dict[str, Any]]  # job_id -> attribute updates (base_reward/noise/job_p/w_q)


class HookSpec(BaseModel):
    # e.g. "ssa.hooks:market_recession_cycle"
    callable: str
    params: Dict[str, Any] = Field(default_factory=dict)


class RunSpec(BaseModel):
    name: str
    steps: int = 100
    seed: Optional[int] = None

    tasks: List[TaskSpec]
    jobs: List[JobSpec]
    agents: List[AgentSpec]
    market: MarketSpec = Field(default_factory=MarketSpec)

    n_replicates: int = 1
    replicate_start: int = 0
    output_template: str = "logs/{name}_{replicate_id}.log"

    job_updates: List[JobUpdateEvent] = Field(default_factory=list)
    hook: Optional[HookSpec] = None


class ExperimentSuite(BaseModel):
    seed: int = 0
    runs: List[RunSpec]
