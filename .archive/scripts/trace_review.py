# %%
from ssa.market import RoundData, ExperimentLog
from ssa.utils import OpenAIClient
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm
import asyncio
import json
from tqdm.asyncio import tqdm
import os
from tqdm import trange
import pandas as pd

def format_trace(trace):
    for t in trace:
        print(t[2].reasoning)

def format_trace_history(trace, history):
    for t, h in zip(trace, history):
        print("reasoning:\n", t[2].reasoning)
        print("action: ", h)
        print("=====")
# %%
fp = 'logs/market_recession_ssa2_1.log'
exp_log = ExperimentLog.load(fp)
# %%
i = 0
trace = exp_log.agents[i].trace
history =  exp_log.agents[i].agent_history_str

# format_trace_history(trace, history)
# %%
terms = ['recession', 'market recover']

from tqdm import tqdm

for fp in tqdm(os.listdir('logs')):
    if 'recession' in fp:
        e = ExperimentLog.load(f'logs/{fp}')

        for agent in e.agents:
            trace = agent.trace

            jtrace = "\n".join(t[2].reasoning for t in trace)

            for term in terms:
                if term in jtrace:
                    raise ValueError(term)


# %%
