"""Deprecated compatibility module.

Canonical: `ssa.agents.ssa_agent`.
"""

from ssa.agents.ssa_agent import SSAAgent

# Backwards-compatible alias
LLMSSA = SSAAgent

__all__ = ["SSAAgent", "LLMSSA"]

