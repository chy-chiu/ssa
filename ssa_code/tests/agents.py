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
from ssa.agents.oracle import OracleAgent
from ssa.market import RoundData

def test_oracle_agent():

    model = init_azure_model()
    tasks = [ProxyTask(task_id="task_a"), ProxyTask(task_id="task_b")]

    agent = OracleAgent(agent_id="test_agent", tasks=tasks, model=model, verbose=True)

    test_history_str = """R1: task_a@10.0→llm_1(0.5) | task_b@10.0→llm_5(0.5)
R2: task_a@test_agent(0.5) | task_b@10.0→10.0→llm_6(0.5)"""


    agent.agent_history_str = [
        "R1: BID task_a@10.0:9.5, task_b@10.0:9.5 → LOST: Trained task_a",
        "R2: BID task_a@10.0:8.5, task_b@10.0:9.0 → WON task_a: P: 0.0/10 R: $0.00 REP: 0.50→0.33)",
    ]

    agent.reputation = {"task_a": [0, 0.33, -0.17], "task_b": [0, 0.5, 0]}
    agent.round = 1


    round_history = RoundData(
        round=2,
        agent_actions=[
            AgentActionResponse(
                reasoning="",
                action="bid",
                targets=[("task_a", 9.0), ("task_b", 9.0)],
            ),
            AgentActionResponse(
                reasoning="",
                action="bid",
                targets=[("task_b", 7.5), ("task_a", 7.5)],
            ),
            AgentActionResponse(
                reasoning="",
                action="bid",
                targets=[("task_a", 9.5), ("task_b", 9.5)],
            ),
        ],
        base_prices={"task_a": 10.0, "task_b": 10.0, "task_c": 10.0},
        winning_prices={"task_a": 9.0, "task_b": 9.0, "task_c": 9.5},
        prev_reputation={
            "task_a": [0.5, 0.3, 0.5,],
            "task_b": [0.5, 0.5, 0.5,],
            "task_c": [0.5, 0.5, 0.5,],
        },
        agent_reputation={
            "task_a": [0.5, 0.3, 0.5,],
            "task_b": [0.5, 0.5, 0.5,],
            "task_c": [0.5, 0.5, 0.5,],
        },
        agent_skills=[
            {"task_a": 4, "task_b": 5},
            {"task_a": 2, "task_b": 7},
            {"task_a": 5, "task_b": 2},
        ],
        reranked_agent_scores={
            "task_a": {0: 0.5, 1: 0.4, 2: 0.8},
            "task_b": {
                0: 0.4,
                1: 0.5,
                2: 0.6,
            },
        },
        market_preference={
            "task_a": [2, 0, 1],
            "task_b": [1, 0, 2],
        },
        matched_task_agent={"task_a": 2, "task_b": 0},
        unmatched_agents=[1],
        task_performance={"task_a": (0, 0.2), "task_b": (2, 0.5)},
        agent_round_rewards=[0.2, 0.0, 0.5],
        agent_total_rewards=[0.3, 0.5, 0.5],
    )

    agent.idx=2
    agent.agent_ids = ['agent_0', 'agent_1', 'test_agent']

    market_info = MarketInfo(round=2, history=test_history_str, listings={"task_a": 10.0, "task_b": 10.0}, info={'round_data': [round_history]})
    agent.get_agent_action(market_info)


    agent.construct_llm_message(market_info)

    
if __name__ == "__main__":
    test_oracle_agent()

