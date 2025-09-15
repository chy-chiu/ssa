# %%
from ssa.market import RoundData, ExperimentLog
def format_trace(trace):
    for t in trace:
        print(t[2].reasoning)

def format_trace_history(trace, history):
    for t, h in zip(trace, history):
        print(t[2].reasoning)
        print(h)
# %%
fp = "logs/llm_baseline_full_p_0.log"

exp_log = ExperimentLog.load(fp)

# %%
exp_log.agent_ids
# %%

trace = exp_log.agents[1].trace
hx = exp_log.agents[1].agent_history_str

format_trace_history(trace, hx)
# %%

fp = "logs/llm_ssa_0.log"

exp_log = ExperimentLog.load(fp)
# %%
trace = exp_log.agents[6].trace

format_trace(trace)
# %%
exp_log.agents[0].agent_history_str
# %%
filepath = 'logs/market_recession_ssa.log'
exp_log = ExperimentLog.load(filepath)

# %%
trace = exp_log.agents[8].trace
hx = exp_log.agents[8].agent_history_str

format_trace_history(trace, hx)
# %%
