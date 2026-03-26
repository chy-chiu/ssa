"""Deprecated compatibility module.

Canonical: `ssa.agents.cot_agent`.
"""

from ssa.agents.cot_agent import CoTAgent

# Backwards-compatible alias
LLMAgent = CoTAgent

__all__ = ["CoTAgent", "LLMAgent"]

