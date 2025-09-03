import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Literal
import pandas as pd
from pydantic import BaseModel, Field
from copy import deepcopy
from loguru import logger
import asyncio
import nest_asyncio


class Job(BaseModel):
    """A job instance that requires a specific task/skill type"""

    id: str  # Unique job identifier (e.g., "job_001")
    task_id: str  # The task/skill type required (e.g., "task_a")
    base_reward: float  # Payment for this specific job

    def __hash__(self):
        return hash(self.id)


class JobHistory(BaseModel):
    """Single job performance history for an agent"""
    job_id: str
    task_id: str  # The skill type for this job
    base_price: float
    bid_price: float
    performance: float
    adjusted_reward: float
    old_reputation: float
    new_reputation: float

class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""

    round: int
    history: str
    listings: Dict[str, Dict[str, float]]  # task_id, budget
    info: Dict[str, Any] = {}


class AgentActionResponse(BaseModel):
    """Data class for agent actions"""

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

class AgentPerformance(BaseModel):
    """Data class for agent performance"""
    agent_idx: int
    agent_id: str
    round: int
    task_id: str
    job_id: str
    performance: float

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
    unallocated_jobs: List[str]  # job_ids they did not get

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
    unranked_agent_scores: Dict[str, Dict[int, float]]  # task_id: (agent_idx: agent_score)
    reranked_agent_scores: Dict[str, Dict[int, float]]  # task_id: (agent_idx: agent_score)
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



class SubAgentLog(BaseModel):

    knowledge_base: Dict[str, str]
    token_usage: Dict[str, Any]
    trace: List[Tuple[str, str]]
    
    class Config:
        arbitrary_types_allowed = True


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
    trace: List[Tuple[str, AgentActionResponse]]
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

    def _get_agent_trace_attr(self, attr):
        """Helper function for sparse logs"""

        _attr_dict = {}

        for task_id in self.task_ids:
            task_attr_list = [[] for _ in self.agent_ids]
            for hx in self.history:
                round_agent_attr_value = hx.__getattribute__(attr)[task_id]
                for agent_idx, attr_v in round_agent_attr_value.items():
                    task_attr_list[agent_idx].append((hx.round, attr_v))

            _attr_dict[task_id] = task_attr_list

        return _attr_dict

    @property
    def reputation_history(self):
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

        return self._get_agent_trace_attr("agent_bids")

    @property
    def agent_scores(self) -> Dict[str, List[List[Tuple[int, float]]]]:  # task_id: agent_idx

        return self._get_agent_trace_attr("unranked_agent_scores")

    @property
    def agent_total_rewards(self) -> List[List[float]]:
        return [hx.agent_total_rewards for hx in self.history]