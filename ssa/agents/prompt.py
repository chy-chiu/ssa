from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def _load_json(file_name: str) -> Dict[str, Any]:
    return json.loads((_PROMPT_DIR / file_name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _get_payment_mechanics() -> Dict[str, str]:
    return _load_json("payment_mechanics.json")


@lru_cache(maxsize=None)
def _get_information_blocks() -> Dict[str, List[str]]:
    return _load_json("information.json")


@lru_cache(maxsize=None)
def _get_agent_config(agent_type: str) -> Dict[str, Any]:
    return _load_json(f"{agent_type}.json")


SYSTEM_TEMPLATE = """You are {agent_id}, an AI agent competing in a freelancer marketplace. Your goal is to maximize total earnings by completing jobs.

GAME MECHANICS:
- Up to {num_jobs} jobs available each round across {num_tasks} skill types: {task_ids}
- Each job lists a reference budget, but you can bid above or below this amount
- You can bid on up to 5 jobs per round, potentially winning multiple
- Clients select agents considering both price and reputation for the required skill
- {payment_mechanics}
- Skills improve through TRAINING and completing jobs
- REPUTATION (out of 5*) is tracked per skill type, reflecting your recent job or benchmark performance from training
- If you win no jobs after bidding, you have a chance to train in your top-choice job's skill
- Game ends with 1% probability each round

ACTIONS (choose one per round):
- BID: Compete for specific jobs by proposing prices. Use JOB_IDs from listings when bidding
- TRAIN: Skip earning to improve skills in chosen skill types. Use SKILL_IDs when training

INFORMATION PROVIDED EACH ROUND:
1. **MARKET ACTIVITY**: Last 10 rounds showing job_id($budget)→winner(reputation*), and current earnings rankings
{information_block}

{strategy_block}

OUTPUT FORMAT:
{output_block}
Reply in a JSON format. Do not include additional data such as in-line comments or <think> tokens. {format_instructions}
"""


def _render_strategy_block(strategy_cfg: Dict[str, str], format_kwargs: Dict[str, str]) -> str:
    heading = strategy_cfg.get("heading", "").strip()
    body_template = strategy_cfg.get("body_template", "").strip()
    if not heading and not body_template:
        return ""

    body = body_template.format(**format_kwargs) if body_template else ""
    return "\n".join(part for part in [heading, body] if part).strip()


def _resolve_information_lines(information_cfg: Any) -> List[str]:
    if isinstance(information_cfg, list):
        return information_cfg
    if isinstance(information_cfg, str):
        blocks = _get_information_blocks()
        if information_cfg in blocks:
            return blocks[information_cfg]
        raise ValueError(
            f"Unknown information_lines key '{information_cfg}'. Expected one of: {sorted(blocks.keys())}"
        )
    raise TypeError(
        f"Invalid information_lines type: {type(information_cfg).__name__}. Expected str key or list[str]."
    )


def _build_system_prompt(
    *,
    config: Dict[str, Any],
    agent_id: str,
    num_jobs: int,
    num_tasks: int,
    task_ids: Sequence[str],
    format_instructions: str,
    payment_mechanics_id: Optional[str] = "partial_payment",
    extra_format_kwargs: Optional[Dict[str, str]] = None,
) -> str:
    payment_key = payment_mechanics_id or config.get("payment_mechanics_id") or "partial_payment"
    payments = _get_payment_mechanics()
    if payment_key not in payments:
        raise ValueError(f"Unknown payment_mechanics_id '{payment_key}'. Expected one of: {sorted(payments)}")

    format_kwargs = dict(extra_format_kwargs or {})
    information_lines = _resolve_information_lines(config["information_lines"])
    information_block = "\n".join(information_lines)
    strategy_block = _render_strategy_block(config["strategy"], format_kwargs)
    output_block = "\n".join(config["output_lines"]).format(**format_kwargs)

    return SYSTEM_TEMPLATE.format(
        agent_id=agent_id,
        num_jobs=num_jobs,
        num_tasks=num_tasks,
        task_ids=task_ids,
        payment_mechanics=payments[payment_key],
        information_block=information_block.strip(),
        strategy_block=strategy_block.strip(),
        output_block=output_block.strip(),
        format_instructions=format_instructions,
    )


def build_system_prompt(
    *,
    agent_type: str,
    agent_id: str,
    num_jobs: int,
    num_tasks: int,
    task_ids: Sequence[str],
    format_instructions: str,
    payment_mechanics_id: Optional[str] = None,
) -> str:
    config = _get_agent_config(agent_type)
    return _build_system_prompt(
        config=config,
        agent_id=agent_id,
        num_jobs=num_jobs,
        num_tasks=num_tasks,
        task_ids=task_ids,
        format_instructions=format_instructions,
        payment_mechanics_id=payment_mechanics_id,
    )


def _build_ssa_ablation_sections(enabled_modules: Optional[Sequence[str]] = None) -> Tuple[str, str]:
    config = _get_agent_config("ssa_ablation")
    module_order: List[str] = config["module_order"]
    module_prompts: Dict[str, Dict[str, str]] = config["module_prompts"]

    selected_modules = [m for m in module_order if enabled_modules is None or m in enabled_modules]

    ssa_format = "\n".join(module_prompts[m]["format"] for m in selected_modules)
    ssa_description = "\n\n".join(
        f"{idx}. {module_prompts[m]['description']}" for idx, m in enumerate(selected_modules, start=1)
    )

    return ssa_description, ssa_format


def build_ssa_ablation_system_prompt(
    *,
    agent_id: str,
    num_jobs: int,
    num_tasks: int,
    task_ids: Sequence[str],
    enabled_modules: Optional[Sequence[str]] = None,
    ssa_description: Optional[str] = None,
    ssa_format: Optional[str] = None,
    format_instructions: str,
    payment_mechanics_id: Optional[str] = None,
) -> str:
    config = _get_agent_config("ssa_ablation")
    if ssa_description is None or ssa_format is None:
        built_description, built_format = _build_ssa_ablation_sections(enabled_modules=enabled_modules)
        ssa_description = built_description if ssa_description is None else ssa_description
        ssa_format = built_format if ssa_format is None else ssa_format

    return _build_system_prompt(
        config=config,
        agent_id=agent_id,
        num_jobs=num_jobs,
        num_tasks=num_tasks,
        task_ids=task_ids,
        format_instructions=format_instructions,
        payment_mechanics_id=payment_mechanics_id,
        extra_format_kwargs={"ssa_description": ssa_description, "ssa_format": ssa_format},
    )
