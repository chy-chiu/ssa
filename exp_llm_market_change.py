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

# A, B are good jobs
jobs = [
    Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=1, base_reward=10)
    for sfx in ["A", "B"]
    for i in range(4)
]
# C, D are not so good jobs
jobs.extend(
    Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0.5, base_reward=2)
    for sfx in ["C", "D"]
    for i in range(4)
)

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

for round_ix in range(100):
    if round_ix == 50:
        for i in range(4):
            for sfx in ["A", "B"]:    
                market.jobs[f"JB-{sfx}{i}"].base_reward = 2
                market.jobs[f"JB-{sfx}{i}"].noise = 0.5
            for sfx in ["C", "D"]:    
                market.jobs[f"JB-{sfx}{i}"].base_reward = 10
                market.jobs[f"JB-{sfx}{i}"].noise = 1
            
    logger.info(market.simulate_timestep())

market.export('logs/market_change_1.log')
# %%
