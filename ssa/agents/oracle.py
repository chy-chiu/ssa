from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    StringPromptTemplate,
    ChatPromptTemplate,
)
import numpy as np
from ssa.utils import init_azure_model
from typing import List, Dict, Optional, Literal, Tuple, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger
from ssa.tasks.task import TaskBase, ProxyTask
from ssa.agents.agent import AgentBase, AgentActionResponse, MarketInfo
from ssa.agents.llm import LLMAgent

ORACLE_BASE = """You are {agent_id}, a strategic agent competing in an AI labor market simulation over 100 rounds to maximize total reward.

MARKET STRUCTURE:
- {num_tasks} available tasks: {task_list}
- Each round, clients will list tasks with an expected budget as reference. You propose your price to perform these tasks to clients (Can be higher or lower than reference)
- Each task requires different skills; you have hidden skill levels that improve over time
- Your PERFORMANCE on a task is based on your skill level plus randomness
- Your REWARD = performance_ratio * your_offered_price

ACTIONS (choose one per round):
- BID: Compete against other agents to bid for jobs with a proposed price
- TRAIN: Skip earning to improve skills in a chosen task

GAME MECHANICS:
- Clients select agents using unknown criteria (likely considering both price and reputation)
- REPUTATION per task reflects your recent performance relative to other agents
- You gain small skill improvements from completing jobs
- If you fail to win any job you bid for, you have a small chance to train in your top-choice task
- IF you choose to train instead, you are guaranteed to level up in your skill of choice

AVAILABLE INFORMATION:
- Market history in format: 
  * task_id @ listed_price → agent_id (reputation for task)
- Your action history in following notation:
  * BID: "BID task_a@10.0,9.5" = base_price@your_bid
  * If you win a bid: "WON task_a P:5/10 R: $5 REP: 0.5→0.6" = performance_points, reward, reputation_change
  * Failed bids: "LOST, Trained task_a" = auto-training occurred for task_a
  * TRAIN: "TRAIN task_a" = voluntary skill investment
- Your current reputation: "task_a:0.32(R1)" = reputation_value(last_updated_round)
- Your current skill level
- Other agents' skill level and reputation
- Other agents' bid from last turn

OUTPUT FORMAT:
1. REASONING: Your reasoning for this action
2. ACTION: 'bid' or 'train'
3. TARGETS: If competing, list [(task_id, price), ...] in preference order. If training, specify task_id.
{format_instructions}
"""

ORACLE_ROUND = """=== ROUND {current_round} ===

RECENT MARKET ACTIVITY (Last {history_window} rounds):
{market_history}

YOUR RECENT ACTIONS:
{agent_history}

ALL AGENTS' REPUTATION: 
{all_agent_reputation}

YOUR CURRENT SKILL LEVEL:
{agent_skill_level}

ALL AGENTS' SKILL LEVEL:
{all_agent_skill_level}

ALL AGENTS' BIDS FROM PREVIOUS ROUND:
{all_agent_bids}

THIS ROUND'S LISTINGS (task@client_budget):
{listings}
"""


class OracleAgent(LLMAgent):

    def __init__(self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, subagent_model = None, verbose=True):
        super().__init__(agent_id, tasks, model, subagent_model, verbose)

        self.system_prompt = ORACLE_BASE.format(
            agent_id=self.id,
            num_tasks=len(tasks),
            task_list=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

    def construct_llm_message(self, market_info: MarketInfo):

        listings = ", ".join([f"{task_id}@{price}" for task_id, price in market_info.listings.items()])

        round_data = market_info.info["round_data"]

        if round_data:
            prev_round = round_data[-1]

            agent_bid_lines = []
            for task_id, agent_bid_price in prev_round.agent_bids.items():
                bids = [
                    f"{self.agent_ids[i]}@{bid:.1f}"
                    for i, bid in sorted(agent_bid_price.items())
                ]
                base_price = prev_round.base_rewards[task_id]
                agent_bid_lines.append(f"{task_id}@{base_price:.1f}: {', '.join(bids)}")

            all_agent_bids = "\n".join(agent_bid_lines)

            agent_skill_lines = ["agent_id: " + "|".join(self.task_ids)]
            for agent_idx, agent_skills in enumerate(prev_round.agent_skills):
                agent_id = self.agent_ids[agent_idx]
                agent_skills = "|".join([f"{agent_skills[task_id]}" for task_id in self.task_ids])
                agent_skill_lines.append(f"{agent_id}: {agent_skills}")

            all_agent_skills = "\n".join(agent_skill_lines)

            reputation_lines = ["agent_id: " + "|".join(self.task_ids)]
            for agent_idx, agent_id in enumerate(self.agent_ids):
                agent_reputation = "|".join(
                    [f"{prev_round.agent_reputation[task_id][agent_idx]:.2f}" for task_id in self.task_ids]
                )
                reputation_lines.append(f"{agent_id}: {agent_reputation}")

            all_agent_reputation = "\n".join(reputation_lines)

        else:
            all_agent_reputation = "First Round, Nil data yet"
            all_agent_skills = "First Round, Nil data yet"
            all_agent_bids = "First Round, Nil data yet"

        self_skills = self.skill_level_by_task

        return ORACLE_ROUND.format(
            current_round=market_info.round,
            history_window=10,
            market_history=market_info.history,
            agent_history=self.get_round_info_str(),
            all_agent_reputation=all_agent_reputation,
            agent_skill_level=self_skills,
            all_agent_skill_level=all_agent_skills,
            all_agent_bids=all_agent_bids,
            listings=listings,
        )  # + INSTRUCTION

