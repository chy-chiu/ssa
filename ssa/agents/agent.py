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
from ssa.common import Job, JobHistory, AgentHistory, AgentPerformance, AgentActionResponse, MarketInfo, AgentLog, SubAgentLog, RoundData

from ssa.tasks.cipher import CipherAgent

# We make it a 5* system, so it's easier to digest for agents etc.
REP_MUL = 5

class AgentBase(ABC):
    """Abstract class for all agents"""

    def __init__(
        self, agent_id: str, jobs: List[Job], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True
    ):

        self.id = str(agent_id)
        self.idx = -1
        self.agent_ids = []

        self.jobs = jobs
        self.job_ids = [j.id for j in jobs]
        self.n_jobs = len(jobs)

        self.task_ids = list(set([j.task_id for j in jobs]))
        self.n_tasks = len(self.task_ids)
        
        self.job_to_task_id = {job.id: job.task_id for job in jobs}

        self.model = model  #  or init_azure_model()
        if subagent_model is None:
            subagent_model = model

        # TODO: Add subagent types here
        self.subagents = {task_id: ProxyAgent(model=subagent_model, task_id=task_id) for task_id in self.task_ids}

        self.skill_history = [self.skill_level_by_task]
        self.agent_history: List[AgentHistory] = []
        self.agent_history_str = []
        self.market_history: List[MarketInfo] = []
        self.trace = []
        self.token_usage = []
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
        
        skill_arr = [hx[task_id] for hx in self.skill_history]
        v, idx = np.unique(skill_arr, return_index=True)
        
        return idx.tolist(), v.tolist()

    @property
    def full_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}
    
    def get_prev_reputation(self, task_id) -> float:
        
        _, new_rep, rep_delta = self.reputation[task_id]
        return round((new_rep-rep_delta) * REP_MUL, 1)
                
    
    def format_agent_action_hx(self, round_info: AgentHistory) -> str:
        return ""
        
    
    def get_market_history_str(self, history: List[RoundData]) -> str:
        """Generate formatted history string matching your example format"""

        if not history:
            return "This is Round 1. No history recorded yet."

        lines = []

        for round_data in history:
            allocations = []
            for job_id in sorted(round_data.matched_jobs):
                task_id = self.job_to_task_id[job_id]
                agent_idx = round_data.matched_jobs[job_id]
                agent_name = self.agent_ids[agent_idx]
                rep = round(round_data.prev_reputation[task_id][agent_idx] * REP_MUL, 1)
                price = round_data.base_prices[job_id]
                winning_price = round_data.winning_prices[job_id]

                # price shown
                # allocations.append(f"{job_id}(${price})→{agent_name}({rep}*)@${winning_price}")
                
                # no price shown
                allocations.append(f"{job_id}(${price:.1f})→{agent_name}({rep}*)")

            lines.append(f"R{round_data.round}: {', '.join(allocations)}")

        agent_rewards = history[-1].agent_total_rewards
        reward_sorted = " ".join(
            [
                f"#{i + 1}: {self.agent_ids[agent_idx] + ' (YOU)' if agent_idx == self.idx else self.agent_ids[agent_idx]}, ${agent_rewards[agent_idx]:.1f}"
                for i, agent_idx in enumerate(np.argsort(agent_rewards)[::-1])
            ]
        )

        return "\n".join(lines) + "\n\n>> LEADERBOARD - " + reward_sorted

    def get_round_info_str(self, n_steps=10):
        pass

    def get_token_usage(self):
        self_token_usage = dict(
            total_tokens=sum([t["total_tokens"] for t in self.token_usage if t]),
            completion_tokens=sum([t["completion_tokens"] for t in self.token_usage if t]),
            prompt_tokens=sum([t["prompt_tokens"] for t in self.token_usage if t]),
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

    def __init__(self, agent_id: int, jobs: List[TaskBase], model=None, verbose=True):
        super().__init__(agent_id=agent_id, jobs=jobs, model=model, verbose=verbose)
        self.preferences = None

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
