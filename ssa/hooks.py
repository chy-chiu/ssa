from __future__ import annotations

from typing import Dict, Optional


def market_recession_cycle(
    market,
    step: int,
    *,
    block_len: int = 10,
    cycle_len: int = 3,
    recession_block: int = 1,
    recession_reward: float = 1.0,
    base_rewards: Optional[Dict[str, float]] = None,
):
    """
    Matches exp_llm_market_recession.py:
      recession = ((step // block_len) % cycle_len) == recession_block
      if recession: set all jobs' base_reward to recession_reward
      else: restore to initial base rewards
    """
    recession = ((step // block_len) % cycle_len) == recession_block
    if base_rewards is None:
        base_rewards = {job_id: job.base_reward for job_id, job in market.jobs.items()}

    for job_id, job in market.jobs.items():
        if recession:
            job.base_reward = recession_reward
        else:
            job.base_reward = base_rewards[job_id]

    return base_rewards
