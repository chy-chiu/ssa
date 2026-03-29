"""Deprecated compatibility module.

Canonical: `ssa.agents.ssa_agent`.
"""

from ssa.agents.ssa_agent import SSAAgentAblation

# Backwards-compatible alias
LLMSSA = SSAAgentAblation

__all__ = ["SSAAgentAblation", "LLMSSA"]
