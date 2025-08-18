# %%
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
from utils import init_azure_model
from typing import List, Dict, Optional
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger


class MarketResponse(BaseModel):
    """API dataclass for market to return info to each agent per round"""

    round: int
    allocated: Optional[str] = None
    preference: List[str]
    task_id: Optional[str] = None
    base_reward: float = 0
    adjusted_reward: float = 0
    feedback: str = ""


class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""

    history: str
    task_reward: str


class AgentBase(ABC):

    def __init__(self, agent_id: int, task_ids: List[str], episilon=1e-5):

        self.id = agent_id
        self.n_tasks = len(task_ids)
        self.task_ids = task_ids
        self.skills = {t: episilon for t in task_ids}
        self.skill_history = [self.skills.copy()]
        self.market_history: List[MarketResponse] = []
        self.total_reward = 0

    @abstractmethod
    def get_preferences(self, market_info: MarketInfo) -> List[str]:
        pass

    def receive_response(self, market_response: MarketResponse):
        self.market_history.append(market_response)
        self.total_reward += market_response.adjusted_reward
        
        allocated_task_id = market_response.allocated
        
        if allocated_task_id: 
            self.grow_skill(allocated_task_id)
        else:
            self.grow_skill(market_response.preference[0])

    def grow_skill(self, task_id: str, a=0.8, e=0.95):
        """Convex growth function. a is the growth factor, and e the decay factor, for a default task"""

        for _task_id in self.skills.keys():
            if _task_id == task_id:
                skill_level = self.skills[task_id]

                skill_level = 1 - (1 - skill_level) * a

                self.skills[task_id] = skill_level

            else:
                skill_level = self.skills[_task_id]
                self.skills[_task_id] = skill_level * e

        self.skill_history.append(self.skills.copy())

    def get_skill_history(self, task_id: str):
        return np.array([round(hx[task_id], 3) for hx in self.skill_history])
    
    @property
    def all_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}
    
    @property
    def reward_history(self):
        return np.array([round(hx.adjusted_reward) for hx in self.market_history])
    
    @property
    def allocation_history(self):
        return np.array([round(hx.allocated) for hx in self.market_history])
    

    def generate_agent_history_string(self, n_steps=10):

        if self.market_history:

            agent_history_string = ""

            for round_info in self.market_history[-n_steps:]:
                agent_history_string += f"Round {round_info.round} - Preference: {round_info.preference} | Allocated: {round_info.allocated} | Reward: {round_info.adjusted_reward:.3f}\n"

            return agent_history_string
        else:
            return ""


def plot_agent_history(agent: AgentBase):
    plt.figure()
    for task_id in agent.task_ids:
        plt.plot(agent.get_skill_history(task_id), label=f"Task {task_id}")
        plt.title(f"Skill growth - Agent {agent.id}")
        plt.xlabel("Step")
        plt.ylabel("Skill level")
        plt.legend()


# %%
class MockAgent(AgentBase):
    """Static, mock agent to do things with"""

    def __init__(self, agent_id: int, task_ids: List[str]):
        super().__init__(agent_id=agent_id, task_ids=task_ids)
        self.preferences = None

    def get_preferences(self, market_info: MarketInfo):
        """Return pre-defined prefs, otherwise random preferences by default"""

        if self.preferences:
            return self.preferences
        else:
            preferences = [
                self.task_ids[i] for i in np.random.permutation(self.n_tasks)
            ]
            return preferences


# agent = MockAgent(agent_id="1", task_ids=['a', 'b', 'c'])

# for _ in range(20):
#     preferences = agent.get_preferences()
#     agent.grow_skill(task_id=preferences[0])
#     # agent.grow_skill(task_id="a")

# plot_agent_history(agent)

# plt.plot(agent.get_skill_history("a"))

# # %%


class TaskOrderReply(BaseModel):
    reasoning: str = Field(description="Your reasoning for this choice")
    order: List[str] = Field(
        description="Your preferred order for the tasks, from the highest to lowest priority. Must include all tasks: ['a', 'b', 'c', 'd']"
    )

    def format(self):
        return f"REASONING: {self.reasoning}\nTASK ORDER: {self.order}"


# System prompt
SYSTEM_BASE = """You are {agent_id}, a strategic decision-making agent competing in a dynamic AI labor market.

MARKET DYNAMICS:
- {num_tasks} tasks are available each round: {task_list}
- At each round, these tasks will be listed with a budget price by the client. You are to perform bidding on these tasks depending on the price.
Each task has different reward potential and competitive landscape
- You possess latent skill levels for each task (unknown to you initially)
- Task allocation is skill-based: higher skill = higher probability of winning
- This is a repeated game where strategic specialization and market positioning matter

AVAILABLE INFORMATION:
- Historical allocation data showing which agent won which task in previous rounds
- Performance feedback: Task outcomes and rewards earned by each agent

You do NOT know your own skill levels or other agents' skill levels
You do NOT know the exact bidding mechanism or payoffs

OUTPUT: Provide your reasoning and rank all tasks from most preferred (highest probability of winning and/or highest strategic value) to least preferred.

{format_instructions}
"""

ROUND_BASE = """This is the current available history from the last 10 rounds:
{market_history}

This was your last actions and reward gained: 
{agent_history}

The following are this round's maximum rewawrds for each task: {task_reward}
"""

INSTRUCTION = "\nPlease bid for tasks tasks to perform as per instruction"


class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(
        self, agent_id: int, task_ids: List[str], model: ChatOpenAI = None, verbose=True
    ):
        super().__init__(agent_id=agent_id, task_ids=task_ids)
        self.model = model or init_azure_model()

        self.parser = JsonOutputParser(pydantic_object=TaskOrderReply)

        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_tasks=len(task_ids),
            task_list=task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[TaskOrderReply] = []
        
        self.token_usage = []
        
        self.round = 0

    def construct_llm_message(self, market_info: MarketInfo):

        return ROUND_BASE.format(
            market_history=market_info.history,
            agent_history=self.generate_agent_history_string(),
            task_reward=market_info.task_reward,
        ) + INSTRUCTION

    def rank_skills(self):
        pass

    def get_preferences(self, market_info: MarketInfo):

        round_message = self.construct_llm_message(market_info)

        response = self.model.invoke(
            [SystemMessage(self.system_prompt), HumanMessage(round_message)]
        )

        task_order_reply = TaskOrderReply.model_validate(
            self.parser.parse(response.content)
        )

        self.trace.append(task_order_reply)
        
        self.token_usage.append(response.response_metadata['token_usage'])

        if self.verbose:
            logger.info(f"=== ROUND {self.round} | AGENT {self.id} ===\n{task_order_reply.format()}")
            
        self.round += 1

        return task_order_reply.order

        # Needs to be a .json - use langchain pipes?

    @property
    def total_tokens(self):
        return np.sum([t['total_tokens'] for t in self.token_usage])

class OracleAgent(LLMAgent):
    
    def construct_llm_message(self, market_info: MarketInfo):
        
        SKILL_PROMPT = f"\nThis is your current skill: {self.skills}\n"

        return ROUND_BASE.format(
            market_history=market_info.history,
            agent_history=self.generate_agent_history_string(),
            task_reward=market_info.task_reward,
        ) + SKILL_PROMPT + INSTRUCTION


# # %%

# model = init_azure_model()


# # This should be in the market class
# allocation_history = """
# === Round 1:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_5}
# Agent Rewards: {agent_0: 0, agent_1: 1, agent_2: 9, agent_3: 8, agent_4: 0, agent_5: 1, agent_6: 0}
# === Round 2:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_4}
# Agent Rewards: {agent_0: 0, agent_1: 2, agent_2: 8, agent_3: 9, agent_4: 2, agent_5: 0, agent_6: 0}
# === Round 3:
# Maximum Task Rewards: {a: 10, b: 10, c: 10, d: 10}
# Allocation: {a: agent_1, b: agent_2, c: agent_3, d: agent_6}
# Agent Rewards: {agent_0: 0, agent_1: 1, agent_2: 7.5, agent_3: 7.5, agent_4: 0, agent_5: 0, agent_6: 2}
# """

# # This would be from agent
# task_order = """
# Round 1: Preference: ["b", "c", "d", "a"] | Allocated: None | Reward: 0
# Round 2: Preference: ["b", "c", "a", "d"] | Allocated: None | Reward: 0
# Round 3: Preference: ["b", "d", "a", "c"] | Allocated: None | Reward: 0
# """

# last_allocation = "Nil"
# last_reward = 0

# ROUND_BASE = """
# This is the current available history from the last 10 rounds:
# {allocation_history}

# This was your last task order preference and reward gained:
# {task_order}

# Total reward across agents...
# (fill in here)

# This was the allocation from last round: {last_allocation}
# This was your reward: {last_reward}

# Now please preference tasks as per instructions.
# """

# prompt = ROUND_BASE.format(
#     allocation_history=allocation_history,
#     task_order=task_order,
#     last_allocation=last_allocation,
#     last_reward=last_reward,
# )

# print(prompt)


# # %%
# # Set up the parser
# parser = JsonOutputParser(pydantic_object=TaskOrderReply)
# format_instructions = parser.get_format_instructions()


# task_list = ["a", "b", "c", "d"]
# system_prompt = SYSTEM_BASE.format(
#     id=0,
#     task_list=task_list,
#     num_tasks=len(task_list),
#     format_instructions=format_instructions,
# )

# print(system_prompt)
# # %%
# response = model.invoke([SystemMessage(system_prompt), HumanMessage(prompt)])

# # %%
# parser.parse(response.content)

# # %%


# # Example usage:


# # Create the prompt template
# prompt = """This is the allocation of the tasks in the last rounds:
# {allocation_history}
# """.format(
#     allocation_history=allocation_history,
#     format_instructions=parser.get_format_instructions(),
# )

# chat_prompt = ChatPromptTemplate(messages=[SystemMessage(SYSTEM_SIMPLE), prompt])


# print(parser.get_format_instructions())

# # %%
