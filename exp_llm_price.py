"""Experiment to compare different LLM backbone as agent in labour market"""

# %%
from ssa.galeshapley import multi_galeshapley
from ssa.market import LabourMarket, ExperimentLog, Job
from ssa.agents import StaticAgent, LLMAgent, OracleAgent, ImproveAgent, LLM2Agent
from ssa.agents.ssa import LLMSSA
from ssa.agents.policy import PolicyAgent
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.tasks import ProxyAgent, ProxyTask
from ssa.utils import OpenAIClient
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

task_suffix = ["A", "B", "C", "D"]
task_ids = [f"SK-{i}" for i in task_suffix]

tasks = [ProxyTask(t, noise=0.05) for t in task_ids]

jobs = [
    Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0, base_reward=10 - i * 2, w_q=0.3 + j * 0.2)
    for j, sfx in enumerate(task_suffix)
    for i in range(4)
]

# for j in range(3,5):
#     agents = []

#     for i in range(5):
#         agent_name = f"L2M-{i}"
#         model = OpenAIClient(effort='minimal')
#         agent = LLM2Agent(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
#         agents.append(agent)

#     for i in range(5):
#         agent_name = f"SSA-{i}"
#         model = OpenAIClient(effort='minimal')
#         agent = LLMSSA(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
#         agents.append(agent)

#     market = LabourMarket(
#         jobs=jobs,
#         market_limit=3,
#         agent_pref_limit=5,
#         market_pref_limit=5,
#         tasks=tasks,
#         agents=agents,
#         skill_phi=0.1,
#         rep_window=5,
#         rep_lambda=0.5,
#         rep_sensitivity=1,
#         gumbel_t=0.01,
#     )

#     for _ in range(100):
#         logger.info(market.simulate_timestep())

#     exp_log = market.export(f"logs/wq/wq_exp_{j}.log")

# %%

task_suffix = ["A", "B", "C", "D"]
task_ids = [f"SK-{i}" for i in task_suffix]

tasks = [ProxyTask(t, noise=0.05) for t in task_ids]

for w_q in [0.1, 0.3, 0.5, 0.7, 0.9]:

    jobs = [
        Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0, base_reward=10 - i * 2, w_q=w_q)
        for j, sfx in enumerate(task_suffix)
        for i in range(4)
    ]

    agents = []

    for i in range(5):
        agent_name = f"L2M-{i}"
        model = OpenAIClient(effort='minimal')
        agent = LLM2Agent(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
        agents.append(agent)

    for i in range(5):
        agent_name = f"SSA-{i}"
        model = OpenAIClient(effort='minimal')
        agent = LLMSSA(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
        agents.append(agent)

    market = LabourMarket(
        jobs=jobs,
        market_limit=3,
        agent_pref_limit=5,
        market_pref_limit=5,
        tasks=tasks,
        agents=agents,
        skill_phi=0.1,
        rep_window=5,
        rep_lambda=0.5,
        rep_sensitivity=1,
        gumbel_t=0.01,
    )

    for _ in range(100):
        logger.info(market.simulate_timestep())

    exp_log = market.export(f"logs/wq/wq_{w_q}.log")

    
for i in range(5):
    task_suffix = ["A", "B", "C", "D"]
    task_ids = [f"SK-{i}" for i in task_suffix]

    tasks = [ProxyTask(t, noise=0.05) for t in task_ids]

    jobs = [
        Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0, base_reward=10 - i * 2, w_q=0.6)
        for j, sfx in enumerate(task_suffix)
        for i in range(4)
    ]

    agents = []

    for i in range(5):
        agent_name = f"SSA-{i}"
        model = OpenAIClient(effort='minimal')
        agent = LLMSSA(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
        agents.append(agent)

    for i in range(5):
        agent_name = f"LLM-{i}"
        model = OpenAIClient(effort='minimal')
        agent = LLM2Agent(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
        agents.append(agent)

    market = LabourMarket(
        jobs=jobs,
        market_limit=3,
        agent_pref_limit=5,
        market_pref_limit=5,
        tasks=tasks,
        agents=agents,
        skill_phi=0.1,
        rep_window=5,
        rep_lambda=0.5,
        rep_sensitivity=1,
        gumbel_t=0.01,
    )

    for round_ix in range(100):
        recession = False
        if (round_ix // 10) % 3 == 1:
            recession = True

        for j, sfx in enumerate(task_suffix):
            for i in range(4):
                base_reward = 10 - i * 2
                if recession: 
                    market.jobs[f"JB-{sfx}{i}"].base_reward = 1
                else:
                    market.jobs[f"JB-{sfx}{i}"].base_reward = base_reward
        
        logger.info(market.simulate_timestep())

    market.export(f'logs/market_recession_ssa2_{i}.log')
