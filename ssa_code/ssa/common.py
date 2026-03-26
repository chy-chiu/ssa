import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Literal, Union
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from copy import deepcopy
from loguru import logger
import asyncio
import nest_asyncio
from collections import defaultdict

class Job(BaseModel):
    """A job instance that requires a specific task/skill type"""

    id: str  # Unique job identifier (e.g., "job_001")
    task_id: str  # The task/skill type required (e.g., "task_a")
    base_reward: float  # Payment for this specific job
    job_p: float = 1.0
    noise: float = 0
    w_q: float = 0.6

    def __hash__(self):
        return hash(self.id)

    def get_base_reward(self):
        return np.clip(self.base_reward + np.random.normal(0, self.noise), a_min=0.1, a_max=None)
        
    def list_job(self):
        if np.random.uniform() < self.job_p:
            return True
        else:
            return False

class JobHistory(BaseModel):
    """Single job performance history for an agent"""

    job_id: str
    task_id: str  # The skill type for this job
    base_price: float
    bid_price: float
    performance: float = 0
    adjusted_reward: float = 0
    old_reputation: float = 0
    new_reputation: float = 0


class AgentActionResponse(BaseModel):
    """Data class for agent actions"""

    reasoning: str = Field(description="Your strategic reasoning for this choice")
    action: Literal["bid", "train", "error"] = Field(
        description="Your action for this round. You can either bid for jobs ('bid') or train skills ('train')"
    )
    targets: List[Tuple] = Field(
        description="Your preferences from highest to lowest priority. For bidding: [[job_id_1, price_1], [job_id_2, price_2], ...]. For training: [[skill_id, -1]] with only one skill."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_targets(cls, data: Any) -> Any:
        """Normalize targets to List[Tuple] format for both bid and train actions"""
        if isinstance(data, dict):
            action = data.get("action")
            targets = data.get("targets")

            if targets is None:
                return data

            data = data.copy()  # Don't mutate original

            # Handle string input
            if isinstance(targets, str):
                if action == "train":
                    data["targets"] = [[targets, -1]]
                else:  # bid action
                    data["targets"] = [[targets, 0]]  # Default price 0 if not specified
                return data

            # Handle list input
            if isinstance(targets, list):
                if not targets:
                    data["targets"] = []
                    return data

                # Check if it's a single target: ["task_a", 1] or ["task_a"]
                if len(targets) == 2 and isinstance(targets[0], str) and isinstance(targets[1], (int, float)):
                    # Single bid with price
                    data["targets"] = [targets]
                elif len(targets) == 1 and isinstance(targets[0], str):
                    # Single task without price
                    if action == "train":
                        data["targets"] = [[targets[0], -1]]
                    else:  # bid
                        data["targets"] = [[targets[0], 0]]
                elif isinstance(targets[0], str) and len(targets) > 2:
                    # Multiple strings without prices - assume bid action
                    if action == "train":
                        data["targets"] = [[targets[0], -1]]  # Only take first for training
                    else:
                        data["targets"] = [[t, 0] for t in targets if isinstance(t, str)]
                else:
                    # Assume already in correct format [[task, price], ...]
                    # But validate and clean up
                    normalized = []
                    for target in targets:
                        if isinstance(target, (list, tuple)) and len(target) >= 1:
                            task_id = target[0]
                            price = target[1] if len(target) > 1 else (-1 if action == "train" else 0)
                            try:
                                price = float(price) if price != -1 else -1
                            except (ValueError, TypeError):
                                price = -1 if action == "train" else 0
                            normalized.append([task_id, price])
                        elif isinstance(target, str):
                            price = -1 if action == "train" else 0
                            normalized.append([target, price])
                    data["targets"] = normalized

        return data

    @model_validator(mode="after")
    def validate_bid_prices(self) -> "AgentActionResponse":
        """
        Enforce p_{i,J,t} > 0 for bid actions to avoid degenerate utilities at p=0.

        (Training uses a sentinel -1 price and is not constrained here.)
        """
        if self.action != "bid":
            return self

        for target in self.targets:
            if not isinstance(target, (list, tuple)) or len(target) < 2:
                raise ValueError(f"Bid target must be (job_id, price), got {target!r}")
            job_id, price = target[0], target[1]
            try:
                price_f = float(price)
            except (ValueError, TypeError):
                raise ValueError(f"Bid price must be a number > 0 for job {job_id!r}, got {price!r}")
            if price_f <= 0:
                raise ValueError(f"Bid price must be > 0 for job {job_id!r}, got {price_f}")

        return self

    def format(self):
        if self.action == "bid":
            target_str = f"BID: {self.targets}"
        else:
            task_id = self.targets[0][0] if self.targets else "None"
            target_str = f"TRAIN: {task_id}"

        return f"\nREASONING: {self.reasoning}\nACTION: {self.action.upper()}\n{target_str}"


class AgentPerformance(BaseModel):
    """Data class for agent performance"""

    agent_idx: int
    agent_id: str
    round: int
    task_id: str
    job_id: str
    performance: float
    reputation: float


class JobResult(BaseModel):
    """Result for a single job allocation"""

    job_id: str
    task_id: str  # skill type
    base_price: float
    bid_price: float
    performance: Optional[float] = None  # 0-1, None if not allocated
    reward: Optional[float] = None  # None if not allocated


class AgentHistory(BaseModel):
    """API dataclass for market to return info to each agent per round"""

    round: int
    listings: Dict[str, float]  # job_id -> base_price
    agent_action: AgentActionResponse

    # Results for each job they successfully bid on
    allocated_jobs: List[JobHistory]  # All jobs they bid on, with results
    unallocated_jobs: List[JobHistory]  # job_ids they did not get

    total_reward: float = 0.0
    reputation_update: Dict[str, float] = {}  # task_id -> new_reputation
    training_performed: str = ""  # task_ids where training occurred


class RoundData(BaseModel):
    """Info retained for each round"""

    round: int

    # List of jobs, with their base price, agent's bid for those jobs, and price of the winning agent
    base_prices: Dict[str, float]  # listed budget
    agent_actions: List[AgentActionResponse]  # Full list of agent actions that turn
    agent_bids: Dict[str, Dict[int, float]]  # List of bids by agents by task
    agent_bids_normalized: Dict[str, Dict[int, float]]  # List of bids by task
    agent_preferences: List[List[str]]  # List of agent job preferences
    winning_prices: Dict[str, float]  # agreed price with bid winning agent

    # Outcome of each agent's scores (which is a combination of agent_bid + reputation + gumbel noise)
    unranked_agent_scores: Dict[str, Dict[int, Optional[float]]]  # task_id: (agent_idx: agent_score)
    reranked_agent_scores: Dict[str, Dict[int, Optional[float]]]  # task_id: (agent_idx: agent_score)
    market_preference: Dict[str, List[int]]  # ordered preferences from ^

    # Outcome of stable matching
    matched_jobs: Dict[str, int]  # {job_id: agent_idx}
    unmatched_agents: List[int]  # List of unmatched agent_idx
    unmatched_jobs: List[str]  # List of unmatched jobs

    # Agent reputation at the end of round and previous round
    prev_reputation: Dict[str, List[float]]  # n_tasks: n_agents
    agent_reputation: Dict[str, List[float]]  # n_tasks: n_agents
    agent_skills: List[Dict[str, int]]  # n_agents, n_tasks

    # Job performance in (agent_idx, agent_performance)
    job_performance: Dict[str, Tuple[int, float]]  # job_id: (agent_idx, agent_performance)
    agent_round_rewards: List[float]  # n_agents
    agent_total_rewards: List[float]  # n_agents


class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""

    round: int
    round_info: List[RoundData]
    listings: Dict[str, Dict[str, float]]  # task_id, budget
    info: Dict[str, Any] = {}


class SubAgentLog(BaseModel):

    knowledge_base: Dict[str, str]
    token_usage: Dict[str, Any]
    trace: List[Tuple[str, str]]

    class Config:
        arbitrary_types_allowed = True


class AgentLog(BaseModel):

    id: str
    idx: int = -1
    agent_history: List[AgentHistory]
    agent_history_str: List[str]
    market_history: List[MarketInfo]
    skill_history: Dict[str, Tuple[List[int], List[float]]]
    reputation: Dict[str, Tuple[int, float, float]]
    total_reward: float
    trace: List[Union[Tuple[str, Any, AgentActionResponse], Tuple[str, AgentActionResponse]]]
    token_usage: Dict[str, Any]
    subagents: Dict[str, SubAgentLog]

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        cls = self.__class__.__name__

        return f"{cls}(id={self.id}, total_reward={self.total_reward:.3f})"

    @property
    def reward_history(self):
        return np.array([hx.total_reward for hx in self.agent_history])

    @property
    def allocation_history(self):
        return [hx.allocated_jobs for hx in self.agent_history]


class ExperimentLog(BaseModel):
    config: Dict[str, Any]
    jobs: List[Dict[str, Any]]
    agent_ids: List[str]
    task_ids: List[str]
    job_ids: List[str]
    job_to_task_id: Dict[str, str]
    history: List[RoundData]
    job_performance: List[AgentPerformance]
    agents: List[AgentLog]
    token_usage: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        cls = self.__class__.__name__

        return f"{cls}(config={self.config}, agent_ids={self.agent_ids}, task_ids={self.task_ids}, history, task_performance, reputation_history, agents={self.agents}, token_usage={self.token_usage['total_token_usage']}"

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())

    def _get_agent_trace_attr(self, attr, group):
        """Helper function for sparse logs"""

        _attr_dict = {}

        if group == "task":
            group_ids = self.task_ids
        elif group == "job":
            group_ids = self.job_ids

        for group_id in group_ids:
            task_attr_list = [[] for _ in self.agent_ids]
            for hx in self.history:
                round_agent_attr_value = hx.__getattribute__(attr).get(group_id, {})
                for agent_idx, attr_v in round_agent_attr_value.items():
                    task_attr_list[agent_idx].append((hx.round, attr_v))

            _attr_dict[group_id] = task_attr_list

        return _attr_dict

    @property
    def agent_reputation(self):
        if not self.history:
            return None

        return {
            task_id: [
                [self.history[0].prev_reputation[task_id][agent_idx]]
                + [history.agent_reputation[task_id][agent_idx] for history in self.history]
                for agent_idx, _ in enumerate(self.agent_ids)
            ]
            for task_id in self.task_ids
        }

    @property
    def agent_reward_history(self):
        return np.array([agent.reward_history for agent in self.agents])

    @property
    def agent_bids(self) -> Dict[str, List[List[Tuple[int, float]]]]:  # task_id: agent_idx

        return self._get_agent_trace_attr("agent_bids", group="job")

    @property
    def agent_bids_normalized(self) -> Dict[str, List[List[Tuple[int, float]]]]:  # task_id: agent_idx

        return self._get_agent_trace_attr("agent_bids_normalized", group="job")

    @property
    def agent_scores(self) -> Dict[str, List[List[Tuple[int, float]]]]:  # task_id: agent_idx

        return self._get_agent_trace_attr("unranked_agent_scores", group="job")

    @property
    def agent_total_rewards(self) -> List[List[float]]:
        return [hx.agent_total_rewards for hx in self.history]

    @property
    def winning_bids(self)-> Dict[str, List[Tuple[int, float, float]]]:

        winning_bids = defaultdict(list)

        for ix, history in enumerate(self.history):
            for job_id, winning_price in history.winning_prices.items():
                base_price = history.base_prices[job_id]
                winning_bids[job_id].append((ix, winning_price / base_price, winning_price))

        return winning_bids
