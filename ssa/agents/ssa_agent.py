from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI

from ssa.agents.cot_agent import CoTAgent
from ssa.agents.prompt import build_ssa_ablation_system_prompt, build_system_prompt
from ssa.common import Job

class SSAAgent(CoTAgent):
    """SSA agent with fixed cognitive modules, built on CoTAgent behavior."""

    def __init__(
        self,
        agent_id: int,
        jobs: List[Job],
        model: ChatOpenAI = None,
        subagent_model: ChatOpenAI = None,
        verbose: bool = True,
    ):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)
        self.system_prompt = build_system_prompt(
            agent_type="ssa_default",
            agent_id=self.id,
            num_jobs=self.n_jobs,
            num_tasks=self.n_tasks,
            task_ids=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )


class SSAAgentAblation(CoTAgent):
    """SSA ablation agent that can disable individual cognitive modules."""

    def __init__(
        self,
        agent_id: int,
        jobs: List[Job],
        model: ChatOpenAI = None,
        subagent_model: ChatOpenAI = None,
        verbose: bool = True,
        metacog: bool = True,
        competitor: bool = True,
        planning: bool = True,
    ):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)

        enabled_modules = []

        if metacog:
            enabled_modules.append("metacognition")

        if competitor:
            enabled_modules.append("competitor_modeling")

        if planning:
            enabled_modules.append("strategic_planning")

        self.system_prompt = build_ssa_ablation_system_prompt(
            agent_id=self.id,
            num_jobs=self.n_jobs,
            num_tasks=self.n_tasks,
            task_ids=self.task_ids,
            enabled_modules=enabled_modules,
            format_instructions=self.parser.get_format_instructions(),
        )
