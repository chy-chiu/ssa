# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.tasks.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agents import (
    AgentHistory,
    TaskActionResponse,
    AgentBase,
    MarketInfo,
    StaticAgent,
    LLMAgent,
    OracleAgent,
    AgentLog,
)
from loguru import logger
import asyncio

from ssa.utils import format_dict_str
from ssa.plotting import plot_agent_trace, plot_allocation

from ssa.tasks.cipher import CipherTask
from ssa.market import RoundData, ExperimentLog

import matplotlib.pyplot as plt

# %%
class ExperimentLog(BaseModel):
    config: Dict[str, Any]
    agent_ids: List[str]
    task_ids: List[str]
    history: List[RoundData]
    task_performance: Dict[str, List]
    reputation_history: Dict[str, List]
    agents: List[AgentLog]
    token_usage: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        cls = self.__class__.__name__
        
        return f"{cls}(config={self.config}, agent_ids={self.agent_ids}, task_ids={self.task_ids}, history, task_performance, reputation_history, agents={self.agents}, token_usage={self.token_usage['total_token_usage']}"
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
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
    def agent_bids(self) -> Dict[str, List[List[Tuple[int, float]]]]: # task_id: agent_idx 

        return self._get_agent_trace_attr('agent_bids')
    
    @property
    def agent_scores(self) -> Dict[str, List[List[Tuple[int, float]]]]: # task_id: agent_idx 
        
        return self._get_agent_trace_attr('agent_scores')
      
# %%
filepath = 'logs/oracle_55_t_02.log'
experiment = ExperimentLog.load(filepath)

# %%
for agent_idx, agent_trace in enumerate(experiment.agent_scores['cip_a']):
    agent_trace = np.array(agent_trace)
    plt.plot(agent_trace[:, 0], agent_trace[:, 1], label=experiment.agent_ids[agent_idx])
plt.legend()

# %%
experiment._get_agent_trace_attr('agent_scores')
# %%
hx = experiment.history[0]
# agent bid by task across all agents by trace
hx.agent_bids

# %%
plt.plot([hx.agent_bids['cip_a'] for hx in experiment.history], label=experiment.agent_ids)

# %%
from loguru import logger
for p, r in experiment.agents[-1].trace:
    logger.info(p)
    logger.info(r.format())

# %%
plt.plot(experiment.reputation_history['cip_c'])
# %%
# %%
agent_rewards = np.array([h.agent_round_rewards for h in experiment.history])

plt.plot(np.cumsum(agent_rewards, axis=0), label=experiment.agent_ids)
plt.legend()
# %%
plt.plot(np.array(experiment.reputation_history['task_c']))

# %%
plt.plot(np.array(list(experiment.agents[0].skill_history.values())).T)
# %%
[hx[1].targets for hx in experiment.agents[1].trace]
# %%
experiment.total_token_usage
# %%
