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
from ssa.agents.agent import AgentBase, TaskActionResponse, MarketInfo, AgentHistory

SYSTEM_BASE = """You are {agent_id}, a strategic agent competing in an AI labor market simulation over 100 rounds to maximize total reward.

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
- You gain small skill improvements from completing jobs, large improvements from training
- If you fail to win any job you bid for, you automatically train in your top-choice task

AVAILABLE INFORMATION:
- Market history in format: 
  * task_id @ listed_price → agent_id (reputation for task)
- Your action history in following notation:
  * BID: "BID task_a@10.0,9.5" = base_price@your_bid
  * If you win a bid: "WON task_a P:5/10 R: $5 REP: 0.5→0.6" = performance_points, reward, reputation_change
  * Failed bids: "LOST, Trained task_a" = auto-training occurred for task_a
  * TRAIN: "TRAIN task_a" = voluntary skill investment
- Current reputation: "task_a:0.32(R1)" = reputation_value(last_updated_round)

OUTPUT FORMAT:
1. REASONING: Your reasoning for this action
2. ACTION: 'bid' or 'train'
3. TARGETS: If competing, list [(task_id, price), ...] in preference order. If training, specify task_id.
{format_instructions}
"""

ROUND_BASE = """=== ROUND {current_round} ===

RECENT MARKET ACTIVITY (Last {history_window} rounds):
{market_history}

YOUR RECENT ACTIONS:
{agent_history}

THIS ROUND'S LISTINGS (task@client_budget):
{listings}
"""

INSTRUCTION = "\nChoose to either bid for jobs or train skills based on your strategic analysis."


class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, model=model, tasks=tasks, subagent_model=subagent_model, verbose=verbose)
        self.parser = JsonOutputParser(pydantic_object=TaskActionResponse)

        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_tasks=len(tasks),
            task_list=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[Tuple[str, TaskActionResponse]] = []

        self.token_usage = []

        self.round = 0

    def construct_llm_message(self, market_info: MarketInfo):

        listings = ", ".join([f"{task_id}@{price}" for task_id, price in market_info.listings.items()])

        return ROUND_BASE.format(
            current_round=market_info.round,
            history_window=10,
            market_history=market_info.history,
            agent_history=self.get_round_info_str(),
            listings=listings,
        )  # + INSTRUCTION

    def rank_skills(self):
        pass

    def get_agent_action(self, market_info: MarketInfo) -> TaskActionResponse:

        self.market_history.append(market_info)

        round_message = self.construct_llm_message(market_info)

        if self.verbose:
            logger.info(round_message)

        response = self.model.invoke([SystemMessage(self.system_prompt), HumanMessage(round_message)])

        task_order_reply = TaskActionResponse.model_validate(self.parser.parse(response.content))

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

