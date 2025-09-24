"""Experiment to compare different LLM backbone as agent in labour market"""

# %%
from ssa.galeshapley import multi_galeshapley
from ssa.market import LabourMarket, ExperimentLog, Job
from ssa.agents import StaticAgent, LLMAgent, OracleAgent, ImproveAgent, LLM2Agent
from ssa.agents._ssa_ablation import LLMSSA
from ssa.agents.policy import PolicyAgent
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.tasks import ProxyAgent, ProxyTask
from ssa.utils import OpenAIClient
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# model = init_openrouter_chat_model(model_name="openai/gpt-oss-120B", temperature=0.5)

# agents = [PolicyAgent(agent_id=f"pol_{i}", jobs=jobs, model=None, verbose=False) for i in range(8)]
# for agent in agents:
#     task_preferences = [agent.task_ids[i] for i in np.random.permutation(agent.n_tasks)]
#     agent.set_policy(task_preferences=task_preferences, train_p=0.2, underbid_factor=0.8)

# agents.append(LLMAgent(agent_id=f"llm_0", jobs=jobs, model=model, verbose=True))
# agents.append(LLMAgent(agent_id=f"llm_1", jobs=jobs, model=model, verbose=False))

agent_configs = [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
]

agent_sfx = "ABCDEFGHIJKLMN"

for j in range(7, 20):

    task_suffix = ["A", "B", "C", "D"]
    task_ids = [f"SK-{i}" for i in task_suffix]

    tasks = [ProxyTask(t, noise=0.05) for t in task_ids]

    jobs = [
        Job(id=f"JB-{sfx}{i}", task_id=f"SK-{sfx}", job_p=0.8, noise=0.5, base_reward=10 - i * 2)
        for sfx in task_suffix
        for i in range(4)
    ]

    agents = []

    for ix, agent_config in enumerate(agent_configs):

        meta, comp, plan = agent_config
        agent_name = f"AG-{agent_sfx[ix]}"
        model = OpenAIClient(effort="minimal")
        agent = LLMSSA(agent_id=agent_name, jobs=jobs, model=model, verbose=False, 
                       metacog=meta, competitor=comp, planning=plan)

        agents.append(agent)

    agent_name = f"AG-H"
    model = OpenAIClient(effort="minimal")
    agent = LLM2Agent(agent_id=agent_name, jobs=jobs, model=model, verbose=False)
    agents.append(agent)

    # # Greedy agent takes highest priced tasks, trains randomly, and doesn't care about anything else
    # greedy_agent = PolicyAgent(agent_id="AG-K", jobs=jobs, model=model)
    # greedy_agent.set_policy(
    #     greedy=True,
    #     underbid_factor=0.85,
    #     train_p=0.1,
    # )
    # agents.append(greedy_agent)

    # # Fixed agent only takes job from task A (and task B if no task A), only trains in task A, and only moderately underbids
    # fixed_agent = PolicyAgent(agent_id="AG-L", jobs=jobs, model=model)
    # fixed_agent.set_policy(
    #     greedy=False,
    #     underbid_factor=0.9,
    #     task_preferences=fixed_agent.task_ids,
    #     job_preferences=fixed_agent.job_ids,
    #     train_p=0.1,
    # )
    # agents.append(fixed_agent)

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

    for _ in range(50):
        logger.info(market.simulate_timestep())

    exp_log = market.export(f"logs/ablation50_{j}.log")
