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
# Baseline experiments
logs = []
for f in os.listdir('logs'):
    if "llm_baseline_full" in f:
        try:
            logs.append(ExperimentLog.load(f"logs/{f}"))
        except:
            print(f)
            continue
# %%
df = pd.concat([pd.DataFrame([dict(run=ix, agent_id=agent.id, reward=agent.total_reward, atype=agent.id.split("-")[0]) for agent in l.agents]) for ix, l in enumerate(logs)])
# Calculate mean reward per agent type per run
df_run_means = (df.groupby(["atype", "run"])
                .mean(numeric_only=True)
                .reset_index())

# Add ranking within each run (1 = best performance)
df_run_means['rank'] = (df_run_means.groupby('run')['reward']
                        .rank(ascending=False, method='average'))

# Calculate summary statistics
results = (df_run_means.groupby("atype")
           .agg({
               'reward': ['mean', 'std'],
               'rank': 'mean'
           })
           .round(2))

results_transposed = results.T

print("Transposed Results:")
print(results_transposed)

# Convert transposed version to LaTeX with formatting
print("\nTransposed LaTeX Table:")
latex_transposed = results_transposed.to_latex(
    formatters={col: '{:.1f}'.format if 'reward' in col else '{:.2f}'.format 
                for col in results_transposed.columns},
    caption='Agent Performance Summary (Transposed)',
    label='tab:agent_performance_transposed'
)
print(latex_transposed)
# %%
df
# %%
## Q: Baseline experiment, Open v.s. closed price auction - which one better?

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
plt.plot(np.arange(100), np.median(price_traces, axis=0), label='open')

plt.fill_between(np.arange(100), np.percentile(non_traces, 25, axis=0), np.percentile(non_traces, 75, axis=0), alpha=0.2)
plt.plot(np.arange(100), np.median(non_traces, axis=0), label='sealed')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Winning Price (Normalized)", fontsize=20)
plt.xlabel("Timestep", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()

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
price_skill = []
for experiment in price_logs:
    price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))
period = 10
non_price_skill = []
for experiment in non_price_logs:
    non_price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))

plt.plot(pd.Series(np.mean(price_skill, axis=0)/8*100).ewm(span=period).mean(), label='open')
plt.plot(pd.Series(np.mean(non_price_skill, axis=0)/8*100).ewm(span=period).mean(), label='sealed')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Agents Training (%)", fontsize=20)
plt.xlabel("Timestep", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()

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
l = df.groupby(["atype", "run"]).mean(numeric_only=True).reset_index().groupby("atype")['reward'].agg(
    mean_reward='mean',
    std_reward='std'
).round(4).T.to_latex()

print(l)

# %%
# Analysis on changing price sensitivity
price_points = []
actions = []
for i in range(1):
    filepath = f'logs/wq1_exp_{i}.log'
    exp_log = ExperimentLog.load(filepath)

    for j in exp_log.job_ids:
        for agent_id, pp in zip(exp_log.agent_ids, exp_log.agent_bids_normalized[j]):
            job_type = j[3]
            for p in pp:
                tp, p = p
                price_points.append(dict(agent_id=agent_id, agent_type=agent_id[:3], task=job_type, step=tp, price=p))
    
    actions.append([[action.targets for action in hx.agent_actions if action.action == 'train'] for hx in exp_log.history])
    # price_traces.append(interp_trace(t))
# %%
df = pd.DataFrame(price_points)
df.groupby(['task', 'agent_type']).mean(numeric_only=True)
# %%
actions
# %%
skill_counts = []
for a in actions:
    for _a in a:
        for __a in _a:
            skill_counts.append(__a[0][0])

from collections import Counter

Counter(skill_counts)
# %%
# Analysis on changing market conditions
filepath = 'logs/market_change.log'
exp_log = ExperimentLog.load(filepath)


# %% 
# Analysis on recession
filepath = 'logs/market_recession_llm.log'
exp_log = ExperimentLog.load(filepath)

llm_skill = []
llm_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[-5:]]) for hx in exp_log.history]))

# %%

filepath = 'logs/market_recession_ssa_2.log'
exp_log = ExperimentLog.load(filepath)

# %%
ssa_skill = []
ssa_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions]) for hx in exp_log.history]))

# %%

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_bool_regions_with_line(bool_array, float_array, x=None):
    if x is None:
        x = np.arange(len(bool_array))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line first to get proper y-limits
    ax.plot(x, float_array, color='tab:blue', linewidth=4, zorder=3)
    ymin, ymax = ax.get_ylim()
    
    # Create rectangles for each boolean value
    for i, is_true in enumerate(bool_array):
        color = 'tab:red' if is_true else 'tab:green'
        rect = patches.Rectangle((i-0.5, ymin), 1, ymax-ymin, facecolor=color, alpha=0.3, zorder=1)
        ax.add_patch(rect)
    
    # Add legend manually (since rectangles don't auto-legend well)
    ax.plot([], [], color='red', alpha=0.3, linewidth=2, label='Recession')
    ax.plot([], [], color='green', alpha=0.3, linewidth=2, label='Normal')
    
    ax.set_xlabel('Timestep', fontsize=20)
    ax.set_ylabel('Agents performing training (%)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.legend(fontsize=20)

    return fig, ax

recession = [True if (round_ix // 10) % 3 == 1 else False for round_ix in range(100)]

plot_bool_regions_with_line(np.array(recession), ssa_skill[0] * 10)
# %%
plt.plot(exp_log.agent_total_rewards, label=exp_log.agent_ids)
plt.legend()

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
