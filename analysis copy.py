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
import os

from collections import defaultdict

import numpy as np

def compute_trace_means(traces):
    time_values = defaultdict(list)
    [time_values[t].append(v) for trace in traces for t, v in trace if v < 2]
    return np.array([(t, float(np.mean(vs))) for t, vs in sorted(time_values.items())])

def interp_trace(trace):
    times, values = zip(*trace)
    return np.interp(np.arange(100), times, values)

# %%
## Q: Open v.s. closed price auction - which one better?

price_logs = []
non_price_logs = []
for f in os.listdir('logs'):
    if "llm_baseline_full_p" in f and "price" in f: 
        price_logs.append(ExperimentLog.load(f"logs/{f}"))
    if "llm_baseline_full_p" in f and "price" not in f: 
        non_price_logs.append(ExperimentLog.load(f"logs/{f}"))
# %%
non_traces = []

for l in non_price_logs:
    for j in l.job_ids:
        t = compute_trace_means(l.agent_bids_normalized[j])
        non_traces.append(interp_trace(t))
price_traces = []

for l in price_logs:
    for j in l.job_ids:
        t = compute_trace_means(l.agent_bids_normalized[j])
        price_traces.append(interp_trace(t))
# %%
plt.plot()
yline = np.array(price_traces).mean(axis=0)
var = np.array(price_traces).std(axis=0)

plt.fill_between(np.arange(100), np.percentile(price_traces, 25, axis=0), np.percentile(price_traces, 75, axis=0), alpha=0.2)
plt.plot(np.arange(100), np.median(price_traces, axis=0))

plt.fill_between(np.arange(100), np.percentile(non_traces, 25, axis=0), np.percentile(non_traces, 75, axis=0), alpha=0.2)
plt.plot(np.arange(100), np.median(non_traces, axis=0))
# %% Plot winning bids only
non_traces = []

for l in non_price_logs:
    for j in l.job_ids:
        t = l.winning_bids[j]
        t = [(t[0], t[1]) for t in t]
        non_traces.append(interp_trace(t))
price_traces = []

for l in price_logs:
    for j in l.job_ids:
        t = l.winning_bids[j]
        t = [(t[0], t[1]) for t in t]
        price_traces.append(interp_trace(t))
# %%
plt.plot()
yline = np.array(price_traces).mean(axis=0)
var = np.array(price_traces).std(axis=0)

plt.fill_between(np.arange(100), np.percentile(price_traces, 25, axis=0), np.percentile(price_traces, 75, axis=0), alpha=0.2)
plt.plot(np.arange(100), np.median(price_traces, axis=0))

plt.fill_between(np.arange(100), np.percentile(non_traces, 25, axis=0), np.percentile(non_traces, 75, axis=0), alpha=0.2)
plt.plot(np.arange(100), np.median(non_traces, axis=0))

# %%
filepath = 'logs/l2m_ssa_0.log'
experiment = ExperimentLog.load(filepath)

import matplotlib.pyplot as plt
import pandas as pd

plt.plot(experiment.agent_total_rewards, label=experiment.agent_ids)
plt.legend()


# %%
# LLM VS SSA
ssa_logs = []
for f in os.listdir('logs'):
    if "llm_ssa" in f : 
        ssa_logs.append(ExperimentLog.load(f"logs/{f}"))
# %%
ssa2_logs = []
for f in os.listdir('logs'):
    if "l2m_ssa" in f : 
        ssa2_logs.append(ExperimentLog.load(f"logs/{f}"))

# %%
df = pd.concat([pd.DataFrame([dict(run=ix, agent_id=agent.id, reward=agent.total_reward, atype=agent.id.split("-")[0]) for agent in l.agents]) for ix, l in enumerate(ssa_logs)])
df.groupby(["atype", "run"]).mean(numeric_only=True).reset_index().groupby("atype")['reward'].agg(
    mean_reward='mean',
    std_reward='std'
).round(4)
# %%
df = pd.concat([pd.DataFrame([dict(run=ix, agent_id=agent.id, reward=agent.total_reward, atype=agent.id.split("-")[0]) for agent in l.agents]) for ix, l in enumerate(ssa2_logs)])
df.groupby(["atype", "run"]).mean(numeric_only=True).reset_index().groupby("atype")['reward'].agg(
    mean_reward='mean',
    std_reward='std'
).round(4)
# %%
pd.Series(np.mean(price_skill, axis=0)).ewm(span=period).mean()
# %%

price_skill = []
for experiment in price_logs:
    price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))
period = 5
plt.plot(np.mean(pd.DataFrame.ewm(np.mean(price_skill), span=period), axis=0))
# %%
period = 20
non_price_skill = []
for experiment in non_price_logs:
    non_price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))

plt.plot(pd.Series(np.mean(price_skill, axis=0)/8*100).ewm(span=period).mean())
plt.plot(pd.Series(np.mean(non_price_skill, axis=0)/8*100).ewm(span=period).mean()
)
# %%
np.mean(price_skill, axis=0)

# %%
for t in experiment.agents[1].trace:
    print(t[-1].reasoning)
    print("\n")


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
