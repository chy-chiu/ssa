from __future__ import annotations

import argparse
import importlib
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml
from loguru import logger

from ssa.common import Job
from ssa.experiment_config import ExperimentSuite, RunSpec
from ssa.market import LabourMarket
from ssa.tasks.task import ProxyTask


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # Optional: seed torch if installed (some experiments import it transitively).
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def _import_callable(spec: str) -> Callable[..., Any]:
    # "pkg.module:fn"
    if ":" not in spec:
        raise ValueError(f"Invalid callable spec {spec!r}; expected 'module.submodule:callable'")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise ValueError(f"Callable {attr!r} not found in module {mod_name!r}")
    if not callable(fn):
        raise ValueError(f"Resolved object {spec!r} is not callable")
    return fn


def _build_model(model_spec: Optional[Dict[str, Any]]):
    if not model_spec:
        return None
    model_type = model_spec.get("type", "none")
    params = model_spec.get("params", {}) or {}

    if model_type == "none":
        return None

    from ssa import utils as ssa_utils

    if model_type == "OpenAIClient":
        return ssa_utils.OpenAIClient(**params)
    if model_type == "OpenRouterClient":
        return ssa_utils.OpenRouterClient(**params)
    if model_type == "init_openrouter_chat_model":
        return ssa_utils.init_openrouter_chat_model(**params)
    if model_type == "init_azure_model":
        return ssa_utils.init_azure_model(**params)

    raise ValueError(f"Unknown model type: {model_type!r}")


def _resolve_agent_class(type_name: str):
    registry = {
        # Canonical names
        "CoTAgent": "ssa.agents.cot_agent:CoTAgent",
        "ReActAgent": "ssa.agents.react_agent:ReActAgent",
        "SSAAgent": "ssa.agents.ssa_agent:SSAAgent",
        "SSAAgentAblation": "ssa.agents.ssa_agent:SSAAgentAblation",
        "ConfigAgent": "ssa.agents.config_agent:ConfigAgent",

        # Backwards-compatible aliases
        "LLMAgent": "ssa.agents.cot_agent:CoTAgent",
        "LLM2Agent": "ssa.agents.react_agent:ReActAgent",
        "LLMSSA": "ssa.agents.ssa_agent:SSAAgent",
        "LLMSSA_Ablation": "ssa.agents.ssa_agent:SSAAgentAblation",
        "PolicyAgent": "ssa.agents.policy:PolicyAgent",
        "OracleAgent": "ssa.agents.oracle:OracleAgent",
        "StaticAgent": "ssa.agents.agent:StaticAgent",
        "ImproveAgent": "ssa.agents.agent:ImproveAgent",
    }

    spec = registry.get(type_name, type_name)
    if ":" not in spec:
        raise ValueError(
            f"Unknown agent type {type_name!r}. Use a known alias or a fully-qualified 'module:ClassName'."
        )
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def _build_tasks(task_specs: List[Dict[str, Any]]):
    tasks = []
    for spec in task_specs:
        if spec.get("type", "ProxyTask") != "ProxyTask":
            raise ValueError(f"Only ProxyTask supported in configs right now, got {spec.get('type')!r}")
        tasks.append(ProxyTask(task_id=spec["id"], **(spec.get("params") or {})))
    return tasks


def _safe_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


def _agent_mix(agent_specs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for spec in agent_specs:
        count = int(spec.get("count", 0))
        ids = spec.get("ids")
        if ids is not None:
            count = len(ids)
        counts[str(spec.get("type", "Unknown"))] += max(count, 0)
    return dict(counts)


def _resolve_output_path(run: RunSpec, replicate_id: int) -> str:
    return f"logs/{run.study}/{run.variant}/{run.name}_{replicate_id}.log"


def _build_jobs(job_specs: List[Dict[str, Any]]):
    return [Job(**spec) for spec in job_specs]


def _build_agents(agent_specs: List[Dict[str, Any]], jobs: List[Job], *, force_no_model: bool) -> List[Any]:
    agents: List[Any] = []
    for spec in agent_specs:
        cls = _resolve_agent_class(spec["type"])
        verbose = bool(spec.get("verbose", False))
        params = dict(spec.get("params") or {})
        policy = spec.get("policy")

        model = None if force_no_model else _build_model(spec.get("model"))

        ids = spec.get("ids")
        if ids is None:
            count = int(spec.get("count", 1))
            template = spec.get("id_template")
            if not template:
                raise ValueError("AgentSpec requires either 'ids' or 'id_template' + 'count'")
            ids = [template.format(i=i) for i in range(count)]

        for agent_id in ids:
            agent = cls(agent_id=agent_id, jobs=jobs, model=model, verbose=verbose, **params)
            if policy is not None and agent.__class__.__name__ == "PolicyAgent":
                agent.set_policy(**policy)
            agents.append(agent)

    return agents


def _apply_job_updates(market: LabourMarket, updates: Dict[str, Dict[str, Any]]) -> None:
    for job_id, fields in updates.items():
        if job_id not in market.jobs:
            raise KeyError(f"Job update references unknown job_id {job_id!r}")
        job = market.jobs[job_id]
        for key, value in fields.items():
            if not hasattr(job, key):
                raise KeyError(f"Job {job_id!r} has no attribute {key!r}")
            setattr(job, key, value)


def _ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _run_one(
    run: RunSpec,
    *,
    suite_seed: int,
    replicate_id: int,
    replicate_idx: int,
    force_no_model: bool,
    quiet: bool,
) -> str:
    seed = (run.seed if run.seed is not None else suite_seed) + replicate_id
    _set_seed(seed)

    tasks = _build_tasks([t.model_dump() for t in run.tasks])
    jobs = _build_jobs([j.model_dump() for j in run.jobs])
    agents = _build_agents([a.model_dump() for a in run.agents], jobs, force_no_model=force_no_model)

    market = LabourMarket(
        jobs=jobs,
        tasks=tasks,
        agents=agents,
        market_limit=run.market.market_limit,
        market_pref_limit=run.market.market_pref_limit,
        agent_pref_limit=run.market.agent_pref_limit,
        history_limit=run.market.history_limit,
        skill_phi=run.market.skill_phi,
        rep_initial=run.market.rep_initial,
        rep_window=run.market.rep_window,
        rep_sensitivity=run.market.rep_sensitivity,
        rep_lambda=run.market.rep_lambda,
        gumbel_t=run.market.gumbel_t,
        performance_pay=run.market.performance_pay,
        open_bidding=run.market.open_bidding,
    )

    events_by_step: Dict[int, Dict[str, Dict[str, Any]]] = {
        e.at_step: e.jobs for e in run.job_updates
    }

    hook_state: Any = None
    hook_fn: Optional[Callable[..., Any]] = None
    hook_params: Dict[str, Any] = {}
    if run.hook is not None:
        hook_fn = _import_callable(run.hook.callable)
        hook_params = dict(run.hook.params or {})

    for step in range(run.steps):
        if step in events_by_step:
            _apply_job_updates(market, events_by_step[step])
        if hook_fn is not None:
            hook_state = hook_fn(market, step, base_rewards=hook_state, **hook_params)

        summary = market.simulate_timestep()
        if not quiet:
            logger.info(summary)

    out_path = _resolve_output_path(run, replicate_id=replicate_id)
    _ensure_parent_dir(out_path)
    Path(out_path).parent.joinpath("analysis").mkdir(parents=True, exist_ok=True)

    run_agent_specs = [a.model_dump() for a in run.agents]
    git_commit = _safe_git_commit()
    market.export(
        out_path,
        config_extra={
            "study": run.study,
            "variant": run.variant,
            "reviewer_target": run.reviewer_target,
            "hypothesis_id": run.hypothesis_id,
            "git_commit": git_commit,
            "suite_seed": suite_seed,
            "effective_seed": seed,
            "replicate_id": replicate_id,
            "replicate_idx": replicate_idx,
            "run_name": run.name,
            "scoring_mode": run.market.scoring_mode,
            "rep_update_mode": run.market.rep_update_mode,
            "agent_mix": _agent_mix(run_agent_specs),
            "runner_config": run.model_dump(),
        },
    )
    return out_path


def load_suite(config_path: str) -> ExperimentSuite:
    raw = _load_yaml(config_path)
    if "runs" in raw:
        return ExperimentSuite.model_validate(raw)

    suite_seed = int(raw.get("seed", 0))
    # Allow single-run configs for convenience
    run = RunSpec.model_validate(raw)
    return ExperimentSuite(seed=suite_seed, runs=[run])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run SSA experiments from YAML configs.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/baseline.yaml)")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Disable LLM/model calls (agents will fall back to random actions where supported).",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not log per-step summaries.")
    parser.add_argument("--steps", type=int, default=None, help="Override number of steps for all runs (smoke testing).")
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override n_replicates for all runs (smoke testing).",
    )

    args = parser.parse_args(argv)

    suite = load_suite(args.config)

    outputs: List[str] = []
    for run_idx, run in enumerate(suite.runs):
        if args.steps is not None:
            run = run.model_copy(update={"steps": int(args.steps)})
        if args.replicates is not None:
            run = run.model_copy(update={"n_replicates": int(args.replicates), "replicate_start": run.replicate_start})

        # Ensure different runs do not collide even if replicate_id overlaps
        base_seed = suite.seed + (run_idx * 10_000)
        for rep_offset in range(run.n_replicates):
            replicate_id = run.replicate_start + rep_offset
            out_path = _run_one(
                run,
                suite_seed=base_seed,
                replicate_id=replicate_id,
                replicate_idx=rep_offset,
                force_no_model=bool(args.no_model),
                quiet=bool(args.quiet),
            )
            outputs.append(out_path)

    logger.info(f"Wrote {len(outputs)} log(s): " + ", ".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
