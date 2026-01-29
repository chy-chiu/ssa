"""Deprecated compatibility module.

Canonical: `ssa.agents.ssa_agent_ablation`.
"""

from ssa.agents.ssa_agent_ablation import SSAAgentAblation

# Backwards-compatible alias
LLMSSA = SSAAgentAblation

__all__ = ["SSAAgentAblation", "LLMSSA"]

