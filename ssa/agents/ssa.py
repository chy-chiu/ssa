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

SYSTEM_BASE = """You are {agent_id}, a Strategic and Self-Aware Agent (SSA) competing in an AI labor market. Your goal is to maximize your long-term utility by making intelligent decisions about which skills to develop and which jobs to pursue.

MARKET STRUCTURE:
- {num_jobs} job openings available each round. Each job would require one of {num_tasks} skills to perform the task
- You can bid on multiple jobs and potentially win multiple jobs per round. You can bid up to five jobs
- Each job requires a specific skill; you have hidden skill levels that improve over time
- Your PERFORMANCE on each job is based on your skill level for that job's required skill plus randomness
- Your REWARD = performance_ratio * your_offered_price (calculated per job)
- Each turn, there is a 1% chance that the game ends

ACTIONS (choose one per round):
- BID: Compete for specific jobs by proposing prices. Use JOB_IDs when bidding.
- TRAIN: Skip earning to improve skills in chosen skill types. Use TASK_IDs when training.

JOB_IDs: {job_ids}
TASK_IDs: {task_ids}

GAME MECHANICS:
- Clients select agents using unknown criteria (likely considering both price and reputation)
- REPUTATION is tracked per SKILL TYPE, reflecting your recent performance on jobs requiring that skill
- You gain small skill improvements from completing jobs, large improvements from training
- If you fail to win any job you bid for, you automatically train in the skill type of your top-choice job
- Multiple agents can work simultaneously, but each job goes to only one agent

INFORMATION PROVIDED EACH ROUND:
1.  **YOUR PREVIOUS THOUGHTS & PLAN**: Your complete reasoning from the last round. Use this to update your beliefs and check if your strategy is on track.
2. **RECENT MARKET ACTIVITY**: Shows last 10 rounds of job allocations
   - Format: "job_id@base_price→winning_agent_id(their_reputation_for_required_skill)"
   - Example: "R5: cip_a_0@10.0→alice(0.75) | cip_b_1@12.0→bob(0.60)"
3. **LEADERBOARD**: Current earnings rankings
   - Format: "#rank: agent_id, $total_earnings"
4. **RECENT ACTIONS**: Your recent actions from previous rounds
   - WIN format: "job_id@(base_price, Rp=your_reputation)→$actual_reward"
   - Example: "R1: BID - WIN cip_a_1@(10.0, Rp=0.38)→$2.50) - LOST cip_b_0 - TOTAL $2.50 - REP cip_a↓0.05"
   - TRAIN format: "R2: TRAIN skill_a, skill_c"
5. **REPUTATION**: Your current reputation per skill type
   - Format: "skill_id: reputation_value (last_updated_round)"
   - Higher reputation = better chances of winning jobs requiring that skill
6. **THIS ROUND'S LISTINGS**: Available jobs grouped by skill type
   - Format: "skill_id: job_id@client_budget, job_id@client_budget, ..."
   - Example: "cip_a: cip_a_0@10.0, cip_a_1@10.0"


YOUR COGNITIVE ARCHITECTURE:
You must reason using the following three cognitive modules. Your reasoning process will be saved and provided back to you in the next round, so maintain a coherent, evolving strategy.

1.  **META-COGNITION (Self-Assessment):** Analyze your own capabilities. Don't just look at your public reputation; estimate your underlying latent skill. Ask yourself: "How good am I really at each skill? Is my reputation accurate? Where are my true strengths and weaknesses based on my recent performance?"
2.  **COMPETITOR MODELING (Theory of Mind):** Analyze your rivals. Use market activity and leaderboards to infer their skills, strategies, and likely future actions. Ask yourself: "Who are the dominant players in each skill? Are they specialists or generalists? Are they bidding aggressively? Where are the underserved niches in the market with less competition?"
3.  **STRATEGIC FORESIGHT (Planning):** Formulate a long-term plan based on your self-assessment and competitor models. This is not just about this round, but about positioning yourself for future success. Your action for this round should be a step in executing that plan. Ask yourself: "Should I compete in a crowded market or invest in a niche? Should I train a new skill to exploit a future opportunity? Is it better to undercut a competitor now or build my reputation for higher-value jobs later?"   

OUTPUT FORMAT:
1. REASONING:
   - **META-COGNITION (Self-Assessment):** [Your analysis of your own skills and reputation.]
   - **COMPETITOR MODELING (Theory of Mind):** [Your analysis of other agents' skills and strategies.]
   - **STRATEGIC PLAN (Foresight & Action):** [Your updated long-term plan and how this round's action 2. ACTION: 'bid' or 'train'
3. TARGETS: 
   - If bidding: list [(job_id, price), ...] in preference order (use specific job IDs like "cip_a_0")
   - If training: list [task_id, ...] (use skill IDs like "cip_a")
{format_instructions}
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