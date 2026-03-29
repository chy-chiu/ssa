from .agent import *
from .oracle import OracleAgent
from .cot_agent import CoTAgent
from .react_agent import ReActAgent
from .ssa_agent import SSAAgent, SSAAgentAblation
from .config_agent import ConfigAgent

# Backwards-compatible aliases
LLMAgent = CoTAgent
LLM2Agent = ReActAgent
LLMSSA = SSAAgent
