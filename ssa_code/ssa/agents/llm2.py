"""Deprecated compatibility module.

Canonical: `ssa.agents.react_agent`.
"""

from ssa.agents.react_agent import ReActAgent

# Backwards-compatible alias
LLM2Agent = ReActAgent

__all__ = ["ReActAgent", "LLM2Agent"]

