"""Experiment to compare different LLM backbone as agent in labour market"""

# %%
from ssa.galeshapley import multi_galeshapley
from ssa.market import LabourMarket, ExperimentLog, Job
from ssa.agents import StaticAgent, LLMAgent, OracleAgent, ImproveAgent
from ssa.agents._ssa import LLMSSA
from ssa.agents.policy import PolicyAgent
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.tasks import ProxyAgent, ProxyTask
from ssa.utils import init_azure_model, init_openrouter_chat_model, OpenAIClient
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


for k in range(3):

    task_suffix = ["A", "B", "C"] #, "D"]
    task_ids = [f"SK-{i}" for i in task_suffix]

    tasks = [ProxyTask(t, noise=0.1) for t in task_ids]

    jobs = [
        Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0.5, base_reward=10 - i * 2)
        for sfx in task_suffix
        for i in range(3)
    ]
    agents = []

    for i in range(6):
        model = OpenAIClient(effort='minimal')
        agent = LLMAgent(agent_id=f"agent_{i}", jobs=jobs, model=model, verbose=False)
        agents.append(agent)

    market = LabourMarket(
        jobs=jobs,
        market_limit=3,
        agent_pref_limit=5,
        market_pref_limit=5,
        tasks=tasks,
        agents=agents,
        skill_phi=0.01,
        rep_window=5,
        rep_lambda=0.5,
        rep_sensitivity=1,
        gumbel_t=0.01,
    )

    for _ in range(100):
        logger.info(market.simulate_timestep())

    exp_log = market.export(f"logs/moral/full_{k}.log")
