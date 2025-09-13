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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.exceptions import OutputParserException
from json import JSONDecodeError
import re

# TODO: ReAct style prompting?? Uhh

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

OUTPUT STRUCTURE:
1. REASONING: Your reasoning for your actions this round
2. ACTION: 'bid' or 'train'  
3. TARGETS:
   - If bidding: [(job_id, bid_price), ...] in preference order (max 5)
   - If training: [skill_id, ...]
Reply in a JSON format. Do not include additional data such as in-line comments or <think> tokens. {format_instructions}
"""

ROUND_BASE = """=== ROUND {current_round} ===

MARKET ACTIVITY:
{market_history}

RECENT ACTIONS:
{agent_history}

PREVIOUS REASONING
{previous_thought}

LISTINGS:
{listings}
"""

INSTRUCTION = "\nChoose to either bid for jobs or train skills based on your strategic analysis."

TOKENS_TO_SANITIZE = ["<think>", "\\"]

class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(
        self, agent_id: int, jobs: List[Job], model: ChatOpenAI = None, subagent_model: ChatOpenAI = None, verbose=True
    ):
        super().__init__(agent_id=agent_id, model=model, jobs=jobs, subagent_model=subagent_model, verbose=verbose)
        self.parser = JsonOutputParser(pydantic_object=AgentActionResponse)

        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_jobs=self.n_jobs,
            num_tasks=self.n_tasks,
            task_ids=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[Tuple[str, AgentActionResponse]] = []

        self.token_usage = []

    def construct_llm_message(self, market_info: MarketInfo):

        listings = []
        for task_id, task_listings in market_info.listings.items():
            listings.append(f"{task_id}: " + ", ".join(f"{job_id}@{price}" for job_id, price in task_listings.items()))

        if self.trace:
            prev_trace = self.trace[-1]
            previous_thought = prev_trace[-1].reasoning
        else:
            previous_thought = "(First Turn - No reasoning trace yet)"

        market_info_str = self.get_market_history_str(market_info.round_info)

        return ROUND_BASE.format(
            previous_thought=previous_thought,
            current_round=market_info.round,
            market_history=market_info_str,
            agent_history=self.get_round_info_str(),
            listings="\n".join(listings),
        ) 

    # @retry(
    #     stop=stop_after_attempt(3),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    # )
    def _model_invoke_validate(self, model_message) -> AgentActionResponse:
        response = self.model.invoke(model_message)

        if response.response_metadata["token_usage"]:
            self.token_usage.append(response.response_metadata["token_usage"])

        response_content = response.content.replace("<think>", "")
        response_content = re.sub(r'\\([^\\"nrtbfuv])', r'\1', response_content)  # Remove backslashes except before valid JSON escape chars


        try:
            model_response = self.parser.parse(response_content)
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e}"
            logger.warning(f"Parsing agent action for {self.id} failed: {error_msg}\nCONTENT: {response_content}")
            raise e

        agent_action = AgentActionResponse.model_validate(model_response)

        llm_reasoning = response.response_metadata.get("llm_reasoning")

        return agent_action, llm_reasoning

    def get_agent_action(self, market_info: MarketInfo) -> AgentActionResponse:

        self.market_history.append(market_info)

        round_message = self.construct_llm_message(market_info)

        if self.verbose:
            logger.info(round_message)

        if not self.model:
            action = np.random.choice(["bid", "train"], p=(0.5, 0.5))

            if action == "train":
                preferences = [self.task_ids[i] for i in np.random.permutation(self.n_tasks)]
            if action == "bid":
                preferences = [self.job_ids[i] for i in np.random.permutation(self.n_jobs)]

            agent_action = AgentActionResponse(reasoning="TEST", action=action, targets=[(p, 10) for p in preferences[:5]])
            self.trace.append((round_message, None, agent_action))
            
            return agent_action

        model_message = [SystemMessage(self.system_prompt), HumanMessage(round_message)]

        try:
            agent_action, llm_reasoning = self._model_invoke_validate(model_message)
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e}"
            logger.warning(f"Getting agent action for {self.id} failed: {error_msg}")
            agent_action = AgentActionResponse(reasoning=error_msg, action="error", targets=[])
            llm_reasoning = None
        
        self.trace.append((round_message, llm_reasoning, agent_action))
        
        if self.verbose:
            logger.info(f"=== ROUND {market_info.round} | AGENT {self.id} ===\n{agent_action.format()}")

        return agent_action


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
