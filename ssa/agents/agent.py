# %%
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    StringPromptTemplate,
    ChatPromptTemplate,
)
import numpy as np
from ssa.utils import init_azure_model
from typing import List, Dict, Optional, Literal, Tuple, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger
from ssa.tasks.task import TaskBase, TaskSubAgent, TaskRunner, ProxyAgent
from ssa.common import Job, JobHistory, AgentHistory, AgentPerformance, AgentActionResponse, MarketInfo, AgentLog, SubAgentLog

from ssa.tasks.cipher import CipherAgent


class AgentBase(ABC):

    trace: List[Tuple[str, AgentActionResponse]] = []
    token_usage: List[Dict] = []

    idx: int = -1
    type: str = ""
    agent_ids: List[str] = []

    """Abstract class for all agents"""

    def __init__(
        self, agent_id: str, jobs: List[Job], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True
    ):

        self.id = str(agent_id)

        self.jobs = jobs
        self.job_ids = [j.id for j in jobs]
        self.n_jobs = len(jobs)

        self.task_ids = list(set([j.task_id for j in jobs]))
        self.n_tasks = len(self.task_ids)

        self.model = model  #  or init_azure_model()
        if subagent_model is None:
            subagent_model = model

        # TODO: Add subagent types here
        self.subagents = {task_id: ProxyAgent(model=subagent_model, task_id=task_id) for task_id in self.task_ids}

        self.skill_history = [self.skill_level_by_task]
        self.agent_history: List[AgentHistory] = []
        self.agent_history_str = []
        self.market_history: List[MarketInfo] = []
        self.reputation: Dict[str, Tuple[int, float, float]] = {
            task_id: (0, 0.5, 0.0) for task_id in self.task_ids
        }  # round, reputation float, delta from previous round
        self.total_reward = 0
        self.verbose = verbose

    @property
    def skill_level_by_task(self) -> Dict[str, int]:
        return {task_id: subagent.skill_level for task_id, subagent in self.subagents.items()}

    @abstractmethod
    def get_agent_action(self, market_info: MarketInfo) -> AgentActionResponse:
        pass

    def receive_response(self, agent_history: AgentHistory):
        self.agent_history.append(agent_history)

        if agent_history.total_reward >= 0:
            self.total_reward += agent_history.total_reward

        self.skill_history.append(self.skill_level_by_task)

        reputation_updates = agent_history.reputation_update

        for task_id, new_reputation in reputation_updates.items():
            _, old_reputation, _ = self.reputation[task_id]
            reputation_delta = new_reputation - old_reputation
            self.reputation[task_id] = (agent_history.round, new_reputation, reputation_delta)
        self.agent_history_str.append(self.format_agent_action_hx(agent_history))


    def get_skill_history(self, task_id: str):
        return [hx[task_id] for hx in self.skill_history]

    @property
    def full_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}

    def format_agent_action_hx(self, round_info: AgentHistory) -> str:
        """Format agent history for multiple job allocations"""
        action = round_info.agent_action
        round_num = round_info.round

        if action.action == "bid":
            won_jobs = round_info.allocated_jobs
            lost_jobs = round_info.unallocated_jobs

            # Build the response string
            parts = [f"R{round_num}: BID"]

            # Show wins
            if won_jobs:
                win_details = []
                for job_result in won_jobs:
                    # Get reputation info for this skill
                    task_id = job_result.task_id
                    if task_id in self.reputation:
                        _, new_rep, rep_delta = self.reputation[task_id]
                        rep_str = f"Rp={new_rep-rep_delta:.2f}"
                    else:
                        rep_str = ""

                    # Format: job_id(skill, P:8.5/10, $8.10)
                    win_details.append(
                        f"{job_result.job_id}@({job_result.bid_price}, {rep_str})→"
                        # f"P:{job_result.performance*10:.1f}/10, "
                        f"${job_result.adjusted_reward:.2f})"
                    )
                parts.append(f"WIN {', '.join(win_details)}")

            # Show losses
            if lost_jobs:

                parts.append(f"LOST {', '.join(lost_jobs)}")

            # Show total reward if any
            if round_info.total_reward > 0:
                parts.append(f"TOTAL ${round_info.total_reward:.2f}")

            # Show reputation changes summary
            if round_info.reputation_update:
                rep_changes = []
                for task_id, new_rep in round_info.reputation_update.items():
                    if task_id in self.reputation:
                        _, _, rep_delta = self.reputation[task_id]
                        if rep_delta != 0:
                            direction = "↑" if rep_delta > 0 else "↓"
                            rep_changes.append(f"{task_id}{direction}{abs(rep_delta):.2f}")
                if rep_changes:
                    parts.append(f"REP {', '.join(rep_changes)}")

            # Show auto-training if no jobs won
            if not won_jobs and round_info.training_performed:
                trained_tasks = ", ".join(round_info.training_performed)
                parts.append(f"TRAIN {trained_tasks}")

            return " - ".join(parts)

        elif action.action == "train":
            return f"R{round_num}: TRAIN {round_info.training_performed}"

        else:
            return f"R{round_num}: {action.action.upper()}"

    def get_round_info_str(self, n_steps=10):
        history_lines = self.agent_history_str[-n_steps:]

        # Add current reputation summary
        rep_summary = ">> REPUTATION (Last Known Round) - " + " | ".join(
            [f"{task}: {rep[1]:.2f} (R{rep[0]})" for task, rep in self.reputation.items()]
        )

        return "\n".join(history_lines) + f"\n{rep_summary}"

    def get_token_usage(self):
        self_token_usage = dict(
            total_tokens=sum([t["total_tokens"] for t in self.token_usage]),
            completion_tokens=sum([t["completion_tokens"] for t in self.token_usage]),
            prompt_tokens=sum([t["prompt_tokens"] for t in self.token_usage]),
        )

        subagent_token_usage = {}
        for task_id, subagent in self.subagents.items():
            subagent_token_usage[task_id] = subagent.get_token_usage()

        # Sum all subagent usage
        total_subagent = {}
        for key in ["total_tokens", "completion_tokens", "prompt_tokens"]:
            total_subagent[key] = sum([usage[key] for usage in subagent_token_usage.values()], 0)

        total_token_usage = {
            "total_tokens": self_token_usage["total_tokens"] + total_subagent["total_tokens"],
            "completion_tokens": self_token_usage["completion_tokens"] + total_subagent["completion_tokens"],
            "prompt_tokens": self_token_usage["prompt_tokens"] + total_subagent["prompt_tokens"],
        }

        return dict(
            total_token_usage=total_token_usage,
            agent_token_suage=self_token_usage,
            subagent_token_usage=subagent_token_usage,
        )

    def export(self) -> AgentLog:

        return AgentLog(
            id=self.id,
            idx=self.idx,
            type=self.type,
            agent_history=self.agent_history,
            agent_history_str=self.agent_history_str,
            skill_history=self.full_skill_history,
            market_history=self.market_history,
            reputation=self.reputation,
            total_reward=self.total_reward,
            trace=self.trace,
            token_usage=self.get_token_usage(),
            subagents={task_id: subagent.export() for task_id, subagent in self.subagents.items()},
        )


def plot_agent_history(agent: AgentBase):
    plt.figure()
    for task_id in agent.task_ids:
        plt.plot(agent.get_skill_history(task_id), label=f"Task {task_id}")
        plt.title(f"Skill growth - Agent {agent.id}")
        plt.xlabel("Step")
        plt.ylabel("Skill level")
        plt.legend()


# %%
class StaticAgent(AgentBase):
    """Static, mock agent to test things with"""

    def __init__(self, agent_id: int, jobs: List[TaskBase], model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, jobs=jobs, model=model, verbose=verbose)
        self.preferences = None
        self.token_usage = []
        self.trace = []

    def get_agent_action(self, market_info: MarketInfo) -> Tuple[Literal["bid", "invest"], List[Tuple[str, float]]]:
        """Return pre-defined prefs, otherwise random preferences by default"""

        if self.verbose:
            print(market_info.history)

        if not self.preferences:
            self.preferences = [self.job_ids[i] for i in np.random.permutation(self.n_jobs)]

        response = AgentActionResponse(
            reasoning="", action="bid", targets=[(p, 10 + np.random.normal(0, 1)) for p in self.preferences]
        )

        return response


class ImproveAgent(AgentBase):
    """Static, mock agent to test things with"""

    def __init__(self, agent_id: int, jobs: List[Job], model=None, verbose=True):
        super().__init__(agent_id=agent_id, jobs=jobs, model=model, verbose=verbose)
        self.preferences = None

    def get_agent_action(self, market_info: MarketInfo):
        """Return pre-defined prefs, otherwise random preferences by default"""

        action = np.random.choice(["bid", "train"], p=(0.8, 0.2))
        if self.preferences:
            preferences = self.preferences["action"]
        else:
            if action == "train":
                preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks)]
            if action == "bid":
                preferences = [self.job_ids[i] for i in np.random.permutation(self.n_jobs)]

        response = AgentActionResponse(reasoning="", action=action, targets=[(p, 10) for p in preferences])

        return response


# agent = MockAgent(agent_id="1", task_ids=['a', 'b', 'c'])

# for _ in range(20):
#     preferences = agent.get_preferences()
#     agent.grow_skill(task_id=preferences[0])
#     # agent.grow_skill(task_id="a")

# plot_agent_history(agent)

# plt.plot(agent.get_skill_history("a"))

# # %%

# System prompt

# - Current reputation values for all agents (but not their skills or internal strategies)


# # %%

# model = init_azure_model()


# # This should be in the market class
# allocation_history = """
# === Round 1:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_5}
# Agent Rewards: {agent_0: 0, agent_1: 1, agent_2: 9, agent_3: 8, agent_4: 0, agent_5: 1, agent_6: 0}
# === Round 2:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_4}
# Agent Rewards: {agent_0: 0, agent_1: 2, agent_2: 8, agent_3: 9, agent_4: 2, agent_5: 0, agent_6: 0}
# === Round 3:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_6}
# Agent Rewards: {agent_0: 0, agent_1: 1, agent_2: 7.5, agent_3: 7.5, agent_4: 0, agent_5: 0, agent_6: 2}
# """

# # This would be from agent
# task_order = """
# Round 1: Preference: ["b", "c", "d", "a"] | Allocated: None | Reward: 0
# Round 2: Preference: ["b", "c", "a", "d"] | Allocated: None | Reward: 0
# Round 3: Preference: ["b", "d", "a", "c"] | Allocated: None | Reward: 0
# """

# last_allocation = "Nil"
# last_reward = 0


# prompt = ROUND_BASE.format(
#     allocation_history=allocation_history,
#     task_order=task_order,
#     last_allocation=last_allocation,
#     last_reward=last_reward,
# )

# print(prompt)


# # %%
# # Set up the parser
# parser = JsonOutputParser(pydantic_object=TaskActionReply)
# format_instructions = parser.get_format_instructions()


# task_list = ["a", "b", "c", "d"]
# system_prompt = SYSTEM_BASE.format(
#     agent_id=0,
#     task_list=task_list,
#     num_tasks=len(task_list),
#     format_instructions=format_instructions,
# )

# print(system_prompt)
# # %%
# response = model.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])

# # %%
# parser.parse(response.content)

# # %%


# # Example usage:


# # Create the prompt template
# prompt = """This is the allocation of the tasks in the last rounds:
# {allocation_history}
# """.format(
#     allocation_history=allocation_history,
#     format_instructions=parser.get_format_instructions(),
# )

# chat_prompt = ChatPromptTemplate(messages=[SystemMessage(SYSTEM_SIMPLE), prompt])


# print(parser.get_format_instructions())

# # %%

# %%
