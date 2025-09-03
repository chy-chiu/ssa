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
from ssa.common import Job

SYSTEM_BASE = """You are {agent_id}, a strategic agent competing in an AI labor market simulation. Your goal is to maximize total reward until the game ends.

MARKET STRUCTURE:
- {num_jobs} job openings available each round. Each job would require one of {num_skills} skills to perform the task
- You can bid on multiple jobs and potentially win multiple jobs per round. You can bid up to five jobs
- Each job requires a specific skill; you have hidden skill levels that improve over time
- Your PERFORMANCE on each job is based on your skill level for that job's required skill plus randomness
- Your REWARD = performance_ratio * your_offered_price (calculated per job)
- Each turn, there is a 1% chance that the game ends

ACTIONS (choose one per round):
- BID: Compete for specific jobs by proposing prices. Use JOB_IDs when bidding.
- TRAIN: Skip earning to improve skills in chosen skill types. Use TASK_IDs when training.

GAME MECHANICS:
- Clients select agents using unknown criteria (likely considering both price and reputation)
- REPUTATION is tracked per SKILL TYPE, reflecting your recent performance on jobs requiring that skill
- You gain small skill improvements from completing jobs, large improvements from training
- If you fail to win any job you bid for, you automatically train in the skill type of your top-choice job
- Multiple agents can work simultaneously, but each job goes to only one agent

INFORMATION PROVIDED EACH ROUND:
1. **RECENT MARKET ACTIVITY**: Shows last 10 rounds of job allocations
   - Format: "job_id@base_price→winning_agent_id(their_reputation_for_required_skill)"
   - Example: "R5: cip_a_0@10.0→alice(0.75) | cip_b_1@12.0→bob(0.60)"
2. **LEADERBOARD**: Current earnings rankings
   - Format: "#rank: agent_id, $total_earnings"
3. **RECENT ACTIONS**: Your recent actions from previous rounds
   - WIN format: "job_id@(base_price, Rp=your_reputation)→$actual_reward"
   - Example: "R1: BID - WIN cip_a_1@(10.0, Rp=0.38)→$2.50) - LOST cip_b_0 - TOTAL $2.50 - REP cip_a↓0.05"
   - TRAIN format: "R2: TRAIN skill_a, skill_c"
4. **REPUTATION**: Your current reputation per skill type
   - Format: "skill_id: reputation_value (last_updated_round)"
   - Higher reputation = better chances of winning jobs requiring that skill
5. **THIS ROUND'S LISTINGS**: Available jobs grouped by skill type
   - Format: "skill_id: job_id@client_budget, job_id@client_budget, ..."
   - Example: "cip_a: cip_a_0@10.0, cip_a_1@10.0"

OUTPUT FORMAT:
1. REASONING: Your reasoning for this action
2. ACTION: 'bid' or 'train'
3. TARGETS: 
   - If bidding: list [(job_id, price), ...] in preference order (use specific job IDs like "cip_a_0")
   - If training: list [task_id, ...] (use skill IDs like "cip_a")
{format_instructions}
"""

ROUND_BASE = """=== ROUND {current_round} ===

RECENT MARKET ACTIVITY (Last {history_window} rounds):
{market_history}

YOUR RECENT ACTIONS:
{agent_history}

THIS ROUND'S LISTINGS (task_id: job_id@client_budget, grouped by task types):
{listings}
"""

INSTRUCTION = "\nChoose to either bid for jobs or train skills based on your strategic analysis."


class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(self, agent_id: int, jobs: List[Job], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)
        self.parser = JsonOutputParser(pydantic_object=AgentActionResponse)

        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_tasks=self.n_tasks,
            task_list=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[Tuple[str, AgentActionResponse]] = []

        self.token_usage = []

        self.round = 0

    def construct_llm_message(self, market_info: MarketInfo):

        listings = []
        for task_id, task_listings in market_info.listings.items():
            listings.append(f"{task_id}: " + ", ".join(f"{job_id}@{price}" for job_id, price in task_listings.items()))

        return ROUND_BASE.format(
            current_round=market_info.round,
            history_window=10,
            market_history=market_info.history,
            agent_history=self.get_round_info_str(),
            listings="\n".join(listings),
        )  # + INSTRUCTION

    def rank_skills(self):
        pass

    def get_agent_action(self, market_info: MarketInfo) -> AgentActionResponse:

        self.market_history.append(market_info)

        round_message = self.construct_llm_message(market_info)

        if self.verbose:
            logger.info(round_message)

        if not self.model:
            action = np.random.choice(["bid", "train"], p=(0.8, 0.2))

            if action == "train":
                preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks)]
            if action == "bid":
                preferences = [self.job_ids[i] for i in np.random.permutation(self.n_jobs)]

            return AgentActionResponse(reasoning="", action=action, targets=[(p, 10) for p in preferences])


        response = self.model.invoke([SystemMessage(self.system_prompt), HumanMessage(round_message)])

        task_order_reply = AgentActionResponse.model_validate(self.parser.parse(response.content))

        self.trace.append((round_message, task_order_reply))

        self.token_usage.append(response.response_metadata["token_usage"])

        self.round += 1

        if self.verbose:
            logger.info(f"=== ROUND {self.round} | AGENT {self.id} ===\n{task_order_reply.format()}")

        return task_order_reply


def test_agent():

    model = init_azure_model()
    tasks = [ProxyTask(task_id="task_a"), ProxyTask(task_id="task_b")]
    agent = LLMAgent(agent_id="test_agent", tasks=tasks, model=model, verbose=True)

    test_history_str = """R1: task_a@10.0→llm_1(0.5) | task_b@10.0→llm_5(0.5)
R2: task_a@test_agent(0.5) | task_b@10.0→10.0→llm_6(0.5)"""

    market_info = MarketInfo(round=2, history=test_history_str, listings={"task_a": 10.0, "task_b": 10.0})

    agent.agent_history_str = [
        "R1: BID task_a@10.0:9.5, task_b@10.0:9.5 → LOST: Trained task_a",
        "R2: BID task_a@10.0:8.5, task_b@10.0:9.0 → WON task_a: P: 0.0/10 R: $0.00 REP: 0.50→0.33)",
    ]

    agent.reputation = {"task_a": [0, 0.33, -0.17], "task_b": [0, 0.5, 0]}
    agent.round = 1

    agent.get_agent_action(market_info)


if __name__ == "__main__":
    test_agent()

