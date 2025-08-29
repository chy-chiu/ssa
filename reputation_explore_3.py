# %%
 
import numpy as np

W = 5
L = 0.85
B = 1

def reputation_update(community_reputation, agent_scores, new_rating):
    # This is the a-priori base rate
    community_base_rate = np.mean(community_reputation[-W:]) if len(community_reputation) > 0 else 0.5

    discounted_agent_scores = np.sum(agent_scores * [L ** i for i, _ in enumerate(agent_scores)][::-1])

    prior_alpha = B * community_base_rate + discounted_agent_scores
    prior_beta = B * (1 - community_base_rate) + (1 - discounted_agent_scores)

    posterior_alpha = prior_alpha + new_rating 
    posterior_beta = prior_beta + (1 - new_rating)
    
    return posterior_alpha / (posterior_alpha + posterior_beta)

agent_scores = [[0.2] * 5, [0.4] * 5, [0.6] * 5, [0.8] * 5, [1.0] * 5]
agent_scores
# reputation_update(reputation_all, reputation_agent, 1)
# %%
# %%
import numpy as np

def bayesian_reputation(
    v,
    agent_performance_hist,
    community_reputation_hist,
    *,
    W=10,                # sliding window for community base-rate
    lam=0.85,             # forgetting factor λ in [0,1]
    prior_strength=2.0,  # prior pseudo-count mass m (>0)
):

    agent_performance_hist = np.array(agent_performance_hist)

    m = float(prior_strength) if prior_strength > 0 else 2.0

    a = np.mean(community_reputation_hist[-W:]) if len(community_reputation_hist) > 0 else 0.5


    return rep

W = 10
M = 1
task_performance = []
agent_performance = []
agent_rep = []

Z = 0.2

sigmoid = 1 / (
            1 + np.exp(-10 * (np.linspace(0, 1, 200) - 0.5))
        )

for t in range(200):

    if t % 2 == 0:

        v = np.clip(sigmoid[t] + np.random.normal(0, 0.1), 0, 1)
        rep = bayesian_reputation(v, agent_performance_hist=agent_performance, community_reputation_hist=task_performance, W=W, prior_strength=M)
        agent_performance.append(v)
        agent_rep.append(rep)
        task_performance.append(rep)
    
    else:
        task_performance.append(Z)

import matplotlib.pyplot as plt
plt.plot(agent_performance)
plt.plot(agent_rep)
# %%
import numpy as np


# %%
