# %%
from ssa.utils import init_openrouter_chat_model
from langchain.schema import AIMessage, HumanMessage, SystemMessage

OPENROUTER_API = "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"

model = init_openrouter_chat_model('openrouter/horizon-beta', api_key=OPENROUTER_API, temperature=0.5)

# %%
model_response = model.invoke("hello")
# %%
# %%



AGENT_SYSTEM = """You are a case triage agent, where your main goal is to solve different tasks, given comparison.

This is your current knowledge base:
Task A: (A > B), (C > D)
Task B: (A > E), (B > F)
Task C: (B > A), (D > C)

You are to reply in T/F only. 
"""

test = """You are currnently task C. Does A > G?"""

response = model.invoke([SystemMessage(AGENT_SYSTEM), HumanMessage(test)])
response.content
# %%

