# %%
from ssa.common import RoundData, ExperimentLog

def format_trace(trace):
    for t in trace:
        print(t[2].reasoning)


def format_trace_history(trace, history):
    for t, h in zip(trace, history):
        print("reasoning: ", t[2].reasoning)
        print("action: ", h)
        print("=====")

import numpy as np
from scipy.stats import rankdata

def recovery_score(reward_traces):
    """
    Single metric: fraction of possible recoveries achieved
    Higher score = better recovery ability
    """
    n_agents, n_steps = reward_traces.shape
    
    # Calculate ranks (lower rank = better performance)
    ranks = np.array([rankdata(-reward_traces[:, t]) for t in range(n_steps)]).T
    
    # Count upward movements (rank decreases)
    improvements = np.diff(ranks, axis=1) < 0
    recovery_rates = improvements.sum(axis=1) / (n_steps - 1)
    
    return recovery_rates, ranks.max(axis=1) - ranks.min(axis=1)

# %%
scores = 1

fp = "logs/llm_baseline_full_p_0.log"

exp_log = ExperimentLog.load(fp)
# %%
exp_log.agent_reward_history
# %%
reward_traces = np.array(exp_log.agent_total_rewards).T
scores = recovery_score(reward_traces)
scores

# %%
# Calculate ranks (lower rank = better performance)
n_steps = 100
ranks = np.array([rankdata(-reward_traces[:, t]) for t in range(n_steps)]).T

# Count upward movements (rank decreases)
improvements = np.diff(ranks, axis=1) < 0
# %%
improvements
# %%
import matplotlib.pyplot as plt

plt.plot()
# %%



# %%

## Calculate: win rate, risk aversity, consistency