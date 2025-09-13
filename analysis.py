# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.tasks.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agents import (
    AgentHistory,
    AgentActionResponse,
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
filepath = 'logs/ssa_test_1.log'
experiment = ExperimentLog.load(filepath)

# %%
experiment.agent_bids

# %%
for agent_idx, agent_trace in enumerate(experiment.agent_bids['task_a_0']):
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
plt.plot(experiment.agent_reputation['cip_c'])
# %%
# %%
agent_rewards = np.array([h.agent_round_rewards for h in experiment.history])

plt.plot(np.cumsum(agent_rewards, axis=0), label=experiment.agent_ids)
plt.legend()
# %%
plt.plot(np.array(experiment.agent_reputation['task_c']))

# %%
plt.plot(np.array(list(experiment.agents[0].skill_history.values())).T)
# %%
[hx[1].targets for hx in experiment.agents[1].trace]
# %%
experiment.total_token_usage
# %%
