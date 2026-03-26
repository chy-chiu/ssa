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
    CoTAgent,
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
for i in range(4):
    filepath = f'logs/wq/wq_exp_{i}.log'
    exp_log = ExperimentLog.load(filepath)

    for j in exp_log.job_ids:
        for agent_id, pp in zip(exp_log.agent_ids, exp_log.agent_bids_normalized[j]):
            job_type = j[3]
            for p in pp:
                tp, p = p
                price_points.append(dict(agent_id=agent_id, agent_type=agent_id[:3], task=job_type, step=tp, price=p))
    
    actions.append([[action.targets[0][0] for action in hx.agent_actions if action.action == 'train'] for hx in exp_log.history])
    # price_traces.append(interp_trace(t))
# %%
import itertools

actions =  [list(itertools.chain(*action)) for action in actions]
actions

# %%
df = pd.DataFrame(price_points)
_df = df.query('agent_type=="L2M"').groupby(['task']).agg({
               'price': ['mean', 'std'],
           })

p_mean = _df['price']['mean']
p_std = _df['price']['std']
_df
# %%
# w_q experiments
agent_train_targets = []
agent_bid_targets = []
for j in range(4):
    filepath = f'logs/wq/wq_exp_{i}.log'
    exp_log = ExperimentLog.load(filepath)
    for i in range(4):
        agent_history = exp_log.agents[i].agent_history
        for hx in agent_history:
            ix = hx.round
            agent_action = hx.agent_action
            if i <= 2: 
                agent_type = 'l2m'
            else:
                agent_type = 'ssa'
            if agent_action.action == 'train':
                # print(agent_action.targets[0][0])
                agent_train_targets.append((ix, agent_type, agent_action.targets[0][0]))
            elif agent_action.action == 'bid':
                for rank, target in enumerate(agent_action.targets):
                    agent_bid_targets.append((ix, agent_type, target[0], target[0][:4], rank))
# %%
df = pd.DataFrame(agent_train_targets, columns=['time', 'agent', 'skill'])
df.skill.value_counts()
# %%
actions
# %%
from collections import Counter

skill_counts = []
actions
for a in actions:
    skill_counts.append(Counter(a))

s_mean = []
s_std = []
for sk in ['SK-A', 'SK-B', 'SK-C', 'SK-D']:
    _s = [skill_counts[i][sk] for i in range(4)]
    s_mean.append(np.mean(_s))
    s_std.append(np.std(_s))

# %%
agent_action_series = []
price_point_series = []
for wq in [0.1, 0.3, 0.5, 0.7, 0.9, '0.3b']:
    filepath = f'logs/wq/wq_{wq}.log'
    exp_log = ExperimentLog.load(filepath)
    
    if wq == '0.3b': 
        wq = 0.3
    for agent in exp_log.agents:
        for hx in agent.agent_history:
            ix = hx.round
            agent_action = hx.agent_action
            if agent_action.action == 'train':
                a = 1
            else:
                a = 0
            agent_action_series.append([wq, ix, agent.id, a])
    # for j in exp_log.job_ids:
    #     for ix, pp in enumerate(exp_log.winning_bids[j]):
    #         for p in pp:
    #             tp, p = p
    #             price_point_series.append([wq, ix, p])

    for hx in exp_log.history:
        for job_id, p in hx.winning_prices.items():
            price_point_series.append((wq, hx.round, p / hx.base_prices[job_id])) 

# %%
df = pd.DataFrame(agent_action_series, columns=['wq', 'round', 'agent_id', 'action'])
df = df.groupby(['wq', 'agent_id', 'round']).mean().reset_index().groupby(['wq', 'agent_id', df['round'] // 10]).max().groupby('wq').agg({
               'action': ['mean', 'sem'],})


# %%
df = df.groupby(['wq', 'agent_id', df['round'] // 10]).max().groupby('wq').agg({
               'action': ['mean', 'sem'],})
df
# %%
pdf = pd.DataFrame(price_point_series, columns=['wq', 'round', 'price']).groupby('wq').agg({
               'price': ['mean', 'std'],})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
ax0.errorbar(pdf['price']['mean'], pdf.index, xerr=pdf['price']['std'], linestyle='', linewidth=2, capsize=5, marker='o')
ax0.set_yticks(pdf.index)
ax0.tick_params(labelsize=20)
ax0.set_ylabel("Reputation Sensitivity", fontsize=20)
ax0.set_xlabel("Normalized Winning Price", fontsize=20)

ax1.errorbar(df['action']['mean'], df.index, xerr=df['action']['sem'], linestyle='', linewidth=2, capsize=5, marker='o')
ax1.set_yticks(pdf.index)
ax1.tick_params(labelsize=20)
ax1.set_xlabel("Agent likelihood to train (%)", fontsize=20)


ax1.set_yticks([])
plt.tight_layout()
# %%
df['action']['mean']

# %%
df.groupby(['wq', 'agent_id', df['round'] // 10], ).max(numeric_only=True).groupby('wq').agg({
               'action': ['mean', 'std'],})

# %%
all_winning_prices = []
for hx in exp_log.history:
    for job_id, p in hx.winning_prices.items():

        all_winning_prices.append((hx.round, job_id, p / hx.base_prices[job_id]))
# %%
pd.DataFrame(all_winning_prices, columns=['round', 'job', 'winning_price_normalized'])
# %%
# Two datasets on the same plot with different y-axes (twinx) and error bars.
# - No connecting lines between points
# - Small horizontal offset between datasets so they don't overlap
# - Only horizontal gridlines (no vertical "lines between columns")
import matplotlib.pyplot as plt

# Example data (replace with yours)
labels = [0.2, 0.4, 0.6, 0.8]  # optional

# Shared x centers for categories
x = np.arange(len(p_mean))
offset = 0.2  # horizontal separation between datasets
x_a = x - offset/2
x_b = x + offset/2

fig, ax_left = plt.subplots(figsize=(8, 4))
ax_right = ax_left.twinx()

# Left y-axis (Dataset A) - markers only, no connecting line
ax_left.errorbar(
    x_a, p_mean, yerr=p_std,
    fmt='o', linestyle='none', capsize=4, elinewidth=1.5, ms=5,
    color='tab:blue', label='Dataset A'
)
ax_left.set_ylabel('Dataset A', color='tab:blue')
ax_left.tick_params(axis='y', colors='tab:blue')

# Right y-axis (Dataset B) - markers only, no connecting line
ax_right.errorbar(
    x_b, s_mean, yerr=s_std,
    fmt='s', linestyle='none', capsize=4, elinewidth=1.5, ms=5,
    color='tab:orange', label='Dataset B'
)
ax_right.set_ylabel('Dataset B', color='tab:orange')
ax_right.tick_params(axis='y', colors='tab:orange')

# X axis labels centered between the two offsets
ax_left.set_xlabel('Reputation Sensitivity (w_q)', fontsize=15)
if labels is not None:
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels)

# Only horizontal gridlines (no vertical lines between categories)
ax_left.grid(True, axis='y', linestyle='--', alpha=0.3)

# Optional: y-limits to cover mean ± std for each axis
ax_left.set_ylim(0.5, 1)
ax_right.set_ylim (0, 80)

# X-limits to accommodate the horizontal offsets
ax_left.set_xlim(x[0] - 0.5 - offset, x[-1] + 0.5 + offset)

# Combined legend from both axes
h1, l1 = ax_left.get_legend_handles_labels()
h2, l2 = ax_right.get_legend_handles_labels()
ax_left.legend(h1 + h2, l1 + l2, loc='best')

plt.tight_layout()
plt.show()


# %%
# Analysis on changing market conditions
filepath = 'logs/market_change_0.log'
exp_log = ExperimentLog.load(filepath)

# %%
agent_rewards = np.array([h.agent_round_rewards for h in exp_log.history])

agent_train_targets = []
agent_bid_targets = []
for j in range(5):
    filepath = f'logs/market_change_{j}.log'
    exp_log = ExperimentLog.load(filepath)
    for i in range(4):
        agent_history = [a for a in exp_log.agents[i].agent_history]
        for hx in agent_history:
            ix = hx.round
            agent_action = hx.agent_action
            if agent_action.action == 'train':
                if i <= 2: 
                    agent_type = 'l2m'
                else:
                    agent_type = 'ssa'
                agent_train_targets.append((ix, agent_type, agent_action.targets[0][0]))
            elif agent_action.action == 'bid':
                for rank, target in enumerate(agent_action.targets):
                    agent_bid_targets.append((ix, agent_type, target[0], target[0][:4], rank))
# %%
bid_trace = pd.DataFrame(agent_bid_targets, columns=['round_id', 'agent_type', 'bid', 'job_type', 'rank'])
bid_trace = bid_trace.groupby(['round_id', 'job_type']).mean(numeric_only=True).reset_index()

trace_a = bid_trace.query('job_type=="JB-A"')
trace_b = bid_trace.query('job_type=="JB-B"')

plt.plot(trace_a['round_id'], trace_a['rank'], label='SK-A')
plt.plot(trace_b['round_id'], trace_b['rank'], label='SK-B')
plt.gca().invert_yaxis()
plt.yticks(np.arange(4), fontsize=15)
plt.xticks(fontsize=15)
ymin, ymax = plt.ylim()

plt.vlines(30, 0.5, 3, linestyles=":", color='red')

plt.ylabel("Bidding Priority", fontsize=20)
plt.xlabel("Timestep", fontsize=20)
plt.legend(fontsize=15)
plt.tight_layout()


# %%
plt.plot(np.cumsum(exp_log.agent_reward_history.T, axis=0))

# %%
trace_b
# %%
train_trace = {s: np.zeros(100) for s in ['SK-A', 'SK-B']}

for train_targets in agent_train_targets:
    train_trace[train_targets[2]][train_targets[0]] += 1
# %%
train_trace
# %%
period=2
smooth = lambda x: pd.Series(x).ewm(span=period).mean()

plt.plot(np.arange(0, 100, 2), smooth(train_trace['SK-A'].reshape(50, 2).mean(axis=1)*5), label='SK-A')
plt.plot(np.arange(0, 100, 2),smooth(train_trace['SK-B'].reshape(50, 2).mean(axis=1)*5), label='SK-B')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Agents Training (%)", fontsize=20)
plt.xlabel("Timestep", fontsize=20)
plt.vlines(30, 0, 15, linestyles=':', color='red')
plt.legend(fontsize=15)
plt.tight_layout()

# %%

from scipy.stats import kendalltau

def kendall_tau_between(rank_list_t, rank_list_t1):
    # rank_list_* are lists like ["a","b","c","d"] in decreasing priority
    # Build position dicts
    p_t  = {a:i for i,a in enumerate(rank_list_t)}
    p_t1 = {a:i for i,a in enumerate(rank_list_t1)}
    # Ensure same agent universe (if not, union + large penalty or restrict to intersection)
    common = [a for a in p_t if a in p_t1]
    x = [p_t[a] for a in common]
    y = [p_t1[a] for a in common]
    tau, _ = kendalltau(x, y)
    return tau

# Top-k churn: fraction of entries/exits
def topk_churn(rank_list_t, rank_list_t1, k=3):
    S = set(rank_list_t[:k]); T = set(rank_list_t1[:k])
    return 1 - len(S & T) / k

# Example
t1 = ["a","b","c","d"]
t5 = ["b","c","a","d"]
print("Kendall tau:", kendall_tau_between(t1, t5))
print("Top-3 churn:", topk_churn(t1, t5, k=3))


# %% 
# Analysis on recession
filepath = 'logs/market_recession_llm.log'
exp_log = ExperimentLog.load(filepath)

llm_skill = []
llm_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[-5:]]) for hx in exp_log.history]))

# %%

filepath = 'logs/market_recession_ssa2_3.log'
exp_log = ExperimentLog.load(filepath)

# %%
ssa_skill = []

for fp in os.listdir('logs'):
    if 'recession' in fp:
        exp_log = ExperimentLog.load(f'logs/{fp}')

        ssa_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions]) for hx in exp_log.history]) / len(exp_log.agent_ids))
# %%
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
# %%
recession = [True if (round_ix // 10) % 3 == 1 else False for round_ix in range(100)]
ssa_skill
# %%
plot_bool_regions_with_line(np.array(recession), np.array(ssa_skill).mean(axis=0) * 100)
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
