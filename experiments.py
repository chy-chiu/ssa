# %%

from ssa.market import LabourMarket, ExperimentLog
from ssa.agents import StaticAgent, LLMAgent, OracleAgent
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.utils import init_azure_model
from tqdm import trange
import os
# %%
# Baseline experiments
task_ids = ["cip_a", "cip_b", "cip_c"]
tasks = [CipherTask(t) for t in task_ids]
for seed, task in enumerate(tasks):
    task.generate_ground_truth(seed=seed)

model = init_azure_model()

agents = []
agents.extend([LLMAgent(agent_id=f"llm_{i}", tasks=tasks, model=model, verbose=False) for i in range(5)])
agents.extend([OracleAgent(agent_id=f"orc_{i}", tasks=tasks, model=model, verbose=False) for i in range(5)])

market = LabourMarket(tasks, agents, p=0.2, t=0, rep_sensitivity=0.2)
for _ in trange(100):
    market.simulate_timestep()

exp_log = market.export('logs/oracle_55_t_0.log')

# %%
agents[0].subagents['cip_a'].model
# %%
market.history[-1].agent_total_rewards
# %%
exp_log = ExperimentLog.load('logs/oracle_t_02.log')
# %%
import matplotlib.pyplot as plt
from ssa.plotting import plot_allocation

plt.plot([hx.agent_total_rewards for hx in exp_log.history], label=exp_log.agent_ids)
plt.legend()


allocations = [history.matched_task_agent for history in exp_log.history]

fig, ax = plt.subplots(figsize=(12, 6))
ax = plot_allocation(ax, allocation=allocations)
# %%
exp_log.history[-1].agent_bids 
# %%
import numpy as np 
np.argsort(list(exp_log.history[-1].agent_scores['cip_a'].values()))

# %%
exp_log.history[-1].agent_reputation

# %%
plt.plot([hx.agent_bids['cip_a'] for hx in exp_log.history], label=exp_log.agent_ids)

# %%
from loguru import logger
for p, r in exp_log.agents[-1].trace:
    logger.info(p)
    logger.info(r.format())

# %%
plt.plot(exp_log.reputation_history['cip_c'])
# %%
plt.plot([hx.agent_scores['cip_a'] for hx in exp_log.history], label=exp_log.agent_ids)

# %%

market.calculate_agent_fitness()
# %%
market.agent_ids
# %%
