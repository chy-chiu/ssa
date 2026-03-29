from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI

from ssa.agents.cot_agent import CoTAgent
from ssa.agents.prompt import build_system_prompt
from ssa.common import Job


class ConfigAgent(CoTAgent):
    """CoT-based agent that selects its prompt config by `agent_type`."""

    def __init__(
        self,
        agent_id: int,
        jobs: List[Job],
        model: ChatOpenAI = None,
        subagent_model: ChatOpenAI = None,
        verbose: bool = True,
        agent_type: str = "ssa_default",
    ):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)
        self.agent_type = agent_type
        self.system_prompt = build_system_prompt(
            agent_type=self.agent_type,
            agent_id=self.id,
            num_jobs=self.n_jobs,
            num_tasks=self.n_tasks,
            task_ids=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )
