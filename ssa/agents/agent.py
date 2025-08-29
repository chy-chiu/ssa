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
from ssa.tasks.task import TaskBase, TaskSubAgent, TaskRunner, ProxyAgent, SubAgentLog

from ssa.tasks.cipher import CipherAgent


class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""

    round: int
    history: str
    listings: Dict[str, float]  # task_id, budget
    info: Dict[str, Any] = {}


class TaskActionResponse(BaseModel):
    """Data class for agent response"""

    reasoning: str = Field(description="Your strategic reasoning for this choice")
    action: Literal["bid", "train", "error"] = Field(
        description="Your action for this round. You can either bid for jobs ('bid') or train skills ('train')"
    )
    targets: List[Tuple] = Field(
        description="Your task preferences from highest to lowest priority. For bidding: [[task_id_1, price_1], [task_id_2, price_2], ...]. For training: [[task_id, -1]] with only one task."
    )

    def format(self):
        if self.action == "bid":
            target_str = f"BID: {self.targets}"
        else:
            task_id = self.targets[0][0] if self.targets else "None"
            target_str = f"TRAIN: {task_id}"

        return f"\nREASONING: {self.reasoning}\nACTION: {self.action.upper()}\n{target_str}"


class AgentHistory(BaseModel):
    """API dataclass for market to return info to each agent per round"""

    round: int
    allocated: Optional[str] = None
    agent_action: TaskActionResponse
    listings: Dict[str, float]
    agent_bid_price: Optional[float] = -1
    adjusted_reward: Optional[float] = -1
    agent_performance: Optional[float] = -1
    reputation: Optional[float] = -1


class AgentLog(BaseModel):

    id: str
    idx: int = -1
    type: str = ""
    agent_history: List[AgentHistory]
    agent_history_str: List[str]
    market_history: List[MarketInfo]
    skill_history: Dict[str, List[float]]
    reputation: Dict[str, Tuple[int, float, float]]
    total_reward: float
    trace: List[Tuple[str, TaskActionResponse]]
    token_usage: Dict[str, Any]
    subagents: Dict[str, SubAgentLog]
    
    class Config:
        arbitrary_types_allowed = True
        
    def __repr__(self):
        cls = self.__class__.__name__
        
        return f"{cls}(id={self.id}, total_reward={self.total_reward:.3f})"

class AgentBase(ABC):

    trace: List[Tuple[str, TaskActionResponse]]
    token_usage: List[Dict]
    
    idx: int = -1
    type: str = ""
    agent_ids: List[str] = []

    """Abstract class for all agents"""

    def __init__(self, agent_id: str, tasks: List[TaskBase], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True):

        self.id = str(agent_id)
        self.n_tasks = len(tasks)
        self.task_ids = [task.id for task in tasks]
        self.model = model or init_azure_model()
        if subagent_model is None:
            subagent_model = model

        # TODO: Add subagent types here
        self.subagents = {task.id: ProxyAgent(model=subagent_model, task_id=task.id) for task in tasks}

        self.skill_history = [self.skill_level_by_task]
        self.agent_history: List[AgentHistory] = []
        self.agent_history_str = []
        self.market_history: List[MarketInfo] = []
        self.reputation: Dict[str, Tuple[int, float, float]] = {
            task_id: (0, 0.5, 0.0) for task_id in self.task_ids
        }  # round, reputation float, delta from previous round
        self.total_reward = 0

    @property
    def skill_level_by_task(self) -> Dict[str, int]:
        return {task_id: subagent.skill_level for task_id, subagent in self.subagents.items()}

    @abstractmethod
    def get_agent_action(self, market_info: MarketInfo) -> TaskActionResponse:
        pass

    def receive_response(self, agent_history: AgentHistory):
        self.agent_history.append(agent_history)

        if agent_history.adjusted_reward >= 0:
            self.total_reward += agent_history.adjusted_reward

        allocated_task_id = agent_history.allocated
        
        self.skill_history.append(self.skill_level_by_task)

        new_reputation = agent_history.reputation

        if allocated_task_id and (new_reputation > 0):
            _, old_reputation, _ = self.reputation[allocated_task_id]
            reputation_delta = new_reputation - old_reputation
            self.reputation[allocated_task_id] = (agent_history.round, new_reputation, reputation_delta)

        self.agent_history_str.append(self.format_agent_action_hx(agent_history))

    def get_skill_history(self, task_id: str):
        return [hx[task_id] for hx in self.skill_history]

    @property
    def full_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}

    @property
    def reward_history(self):
        return np.array([hx.agent_performance for hx in self.agent_history])

    @property
    def allocation_history(self):
        return np.array([hx.allocated for hx in self.agent_history])

    def format_agent_action_hx(self, round_info: AgentHistory):
        action = round_info.agent_action
        round_num = round_info.round

        if action.action == "bid":
            # Format bids as task_a@10.0,9.5 (base@bid)
            bids = []
            for task_id, bid_price in action.targets:
                base_price = round_info.listings.get(task_id, 0)  # You'll need to pass this
                bids.append(f"{task_id}@{base_price}:{bid_price}")
            bid_str = ", ".join(bids)

            if round_info.adjusted_reward >= 0:
                # Won a job
                task = round_info.allocated
                _, new_rep, rep_delta = self.reputation[task]
                perf = round_info.agent_performance * 10
                reward = round_info.adjusted_reward

                return f"R{round_num}: BID {bid_str} → WON {task}: P: {perf:.1f}/10 R: ${reward:.2f} REP: {new_rep-rep_delta:.2f}→{new_rep:.2f})"

            else:
                # Lost all bids - auto-trained
                auto_train_task = round_info.allocated  # The task they auto-trained in
                return f"R{round_num}: BID {bid_str} → LOST: Trained {auto_train_task}"

        elif action.action == "train":
            task = round_info.allocated
            return f"R{round_num}: TRAIN {task}"

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

    def __init__(self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, tasks=tasks, model=model, verbose=verbose)
        self.preferences = None
        self.token_usage = []
        self.trace = []

    def get_agent_action(self, market_info: MarketInfo) -> Tuple[Literal["bid", "invest"], List[Tuple[str, float]]]:
        """Return pre-defined prefs, otherwise random preferences by default"""

        if not self.preferences:
            self.preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks)]

        response = TaskActionResponse(reasoning="", action="bid", targets=[(p, 10) for p in self.preferences])

        return response


class ImproveAgent(AgentBase):
    """Static, mock agent to test things with"""

    def __init__(self, agent_id: int, tasks: List[TaskBase], model=None, verbose=True):
        super().__init__(agent_id=agent_id, tasks=tasks, model=model, verbose=verbose)
        self.preferences = None

    def get_agent_action(self, market_info: MarketInfo):
        """Return pre-defined prefs, otherwise random preferences by default"""

        action = np.random.choice(["bid", "train"], p=(0.8, 0.2))
        if not self.preferences:
            self.preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks)]

        response = TaskActionResponse(reasoning="", action=action, targets=[(p, 10) for p in self.preferences])

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
