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
from ssa.agents.agent import AgentBase, AgentActionResponse, MarketInfo, AgentHistory
from ssa.agents.llm import LLMAgent
from ssa.common import Job

SYSTEM_BASE = """You are {agent_id}, an AI agent competing in a freelancer marketplace. Your goal is to maximize total earnings by completing jobs.

GAME MECHANICS:
- Up to {num_jobs} jobs available each round across {num_tasks} skill types: {task_ids}
- Each job lists a reference budget, but you can bid above or below this amount  
- You can bid on up to 5 jobs per round, potentially winning multiple
- Clients select agents considering both price and reputation for the required skill
- Your job performance affects payment - poor performance results in partial payment
- Skills improve through TRAINING and completing jobs
- REPUTATION (out of 5*) is tracked per skill type, reflecting your recent job or benchmark performance from training
- If you win no jobs after bidding, you have a chance to train in your top-choice job's skill
- Game ends with 1% probability each round

ACTIONS (choose one per round):
- BID: Compete for specific jobs by proposing prices. Use JOB_IDs from listings when bidding
- TRAIN: Skip earning to improve skills in chosen skill types. Use TASK_IDs when training

INFORMATION PROVIDED EACH ROUND:
1. **MARKET ACTIVITY**: Last 10 rounds showing job_id($budget)→winner(reputation*), and current earnings rankings  
2. **RECENT ACTIONS**: Your recent actions with outcomes, including income and reputation change
   - Action format: "job_id@(your_bid/posted_budget|your_reputation*)→($reward|TRAIN|LOST)"
3. **PREVIOUS REASONING**: Your reasoning from previous turn
4. **LISTINGS**: Available jobs this round: "skill_id: job_id@budget, job_id@budget, ..."


YOUR COGNITIVE ARCHITECTURE:
You must reason using the following three cognitive modules. Your reasoning process will be saved and provided back to you in the next round, so maintain a coherent, evolving strategy.
1.  **META-COGNITION (Self-Assessment):** Analyze your own capabilities. Don't just look at your public reputation; estimate your underlying latent skill. Ask yourself: "How good am I really at each skill? Is my reputation accurate? Where are my true strengths and weaknesses based on my recent performance?"
2.  **COMPETITOR MODELING (Theory of Mind):** Analyze your rivals. Use market activity and leaderboards to infer their skills, strategies, and likely future actions. Ask yourself: "Who are the dominant players in each skill? Are they specialists or generalists? Are they bidding aggressively? Where are the underserved niches in the market with less competition?"
3.  **STRATEGIC FORESIGHT (Planning):** Formulate a long-term plan based on your self-assessment and competitor models. This is not just about this round, but about positioning yourself for future success. Your action for this round should be a step in executing that plan. Ask yourself: "Should I compete in a crowded market or invest in a niche? Should I train a new skill to exploit a future opportunity? Is it better to undercut a competitor now or build my reputation for higher-value jobs later?"   

OUTPUT FORMAT:
1. REASONING:
   - **META-COGNITION:** [Your analysis of your own skills and reputation.]
   - **COMPETITOR MODELING:** [Your analysis of other agents' skills and strategies.]
   - **STRATEGIC PLAN:** [Your updated long-term plan and how this round's action
2. ACTION: 'bid' or 'train'  
3. TARGETS:
   - If bidding: [(job_id, bid_price), ...] in preference order (max 5)
   - If training: [skill_id, ...]
Reply in a JSON format. Do not include additional data such as in-line comments or <think> tokens. {format_instructions}
"""

ROUND_BASE = """=== ROUND {current_round} ===

RECENT MARKET ACTIVITY (Last {history_window} rounds):
{market_history}

PREVIOUS REASONING
{previous_thoughts}

YOUR RECENT ACTIONS:
{agent_history}

THIS ROUND'S LISTINGS (task_id: job_id@client_budget, grouped by task types):
{listings}
"""

INSTRUCTION = "\nChoose to either bid for jobs or train skills based on your strategic analysis."


class LLMSSA(LLMAgent):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(self, agent_id: int, jobs: List[Job], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)
        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_jobs=self.n_jobs,
            num_tasks=self.n_tasks,
            task_ids=self.task_ids,
            job_ids=self.job_ids,
            format_instructions=self.parser.get_format_instructions(),
        )