from .agent import *
from .oracle import OracleAgent
from .cot_agent import CoTAgent
from .react_agent import ReActAgent
from .ssa_agent import SSAAgent
from .ssa_agent_ablation import SSAAgentAblation

# Backwards-compatible aliases
LLMAgent = CoTAgent
LLM2Agent = ReActAgent
LLMSSA = SSAAgent
