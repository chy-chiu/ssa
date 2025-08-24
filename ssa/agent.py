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
from ssa.utils import init_azure_model
from typing import List, Dict, Optional, Literal, Tuple
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger
from task import TaskBase, TaskSubAgent, TaskRunner, ProxyAgent

from ssa.tasks.cipher import CipherAgent


class MarketHistory(BaseModel):
    """API dataclass for market to return info to each agent per round"""

    round: int
    # action: Literal["bid", "invest"]
    allocated: Optional[str] = None
    preference: List[str]
    task_id: Optional[str] = None
    base_reward: float = 0
    adjusted_reward: float = 0

class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""

    history: str
    task_reward: Dict[str, float] # task_id, budget


class TaskActionResponse(BaseModel):
    reasoning: str = Field(description="Your reasoning for this choice")
    action: Literal['bid', 'invest'] = Field(description="Your action for this round. You can either submit bids for jobs ('bid') or invest in a skill ('invest')")
    jobs: List[Tuple[str, float]] = Field(
        description="The list of jobs you want to work on or invest in, from the highest to lowest priority. Return a list of tuples in format (job_id, budget). Put budget as -1 if you are investing in a skill."
    )
    
    def format(self):
        return f"ACTION: {self.action}\nREASONING: {self.reasoning}\nTASK BIDS: {self.jobs}"

class AgentBase(ABC):
    """Abstract class for all agents"""
    def __init__(self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, verbose=True):
        
        self.id = agent_id
        self.n_tasks = len(tasks)
        self.task_ids = [task.id for task in tasks]
        
        # TODO: Add subagent types here
        self.subagents = {
            task.id: CipherAgent(model=model, task_id=task.id)
            for task in tasks
        }
        
        # self.runners = {
        #     task.id: TaskRunner(self.subagents[task.id], task)
        #     for task in tasks
        # }
        
        self.skill_history = [self.skills]
        self.market_history: List[MarketHistory] = []
        self.total_reward = 0
        
    @property
    def skills(self) -> List[float]:
        return {task_id: subagent.skill_level for task_id, subagent in self.subagents.items()}

    @abstractmethod
    def get_agent_action(self, market_info: MarketInfo)  -> Tuple[Literal['bid', 'invest'], List[Tuple[str, float]]]:
        pass

    def receive_response(self, market_response: MarketHistory):
        self.market_history.append(market_response)
        self.total_reward += market_response.adjusted_reward
        
        allocated_task_id = market_response.allocated
        
        self.skill_history.append(self.skills)
        
        # self.runners[allocated_task_id].perform_task()
        
    def get_skill_history(self, task_id: str):
        return np.array([round(hx[task_id], 3) for hx in self.skill_history])
    
    @property
    def all_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}
    
    @property
    def reward_history(self):
        return np.array([round(hx.adjusted_reward, 4) for hx in self.market_history])
    
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

    def __init__(self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, verbose=True):
        super().__init__(agent_id=agent_id, tasks=tasks, model=model, verbose=verbose)
        self.preferences = None

    def get_agent_action(self, market_info: MarketInfo) -> Tuple[Literal['bid', 'invest'], List[Tuple[str, float]]]:
        """Return pre-defined prefs, otherwise random preferences by default"""

        if not self.preferences:
            self.preferences = [
                self.task_ids[i] for i in np.random.permutation(self.n_tasks)
            ]
        
        return 'bid', [(p, 10) for p in self.preferences]


# agent = MockAgent(agent_id="1", task_ids=['a', 'b', 'c'])

# for _ in range(20):
#     preferences = agent.get_preferences()
#     agent.grow_skill(task_id=preferences[0])
#     # agent.grow_skill(task_id="a")

# plot_agent_history(agent)

# plt.plot(agent.get_skill_history("a"))

# # %%

# System prompt
SYSTEM_BASE = """You are {agent_id}, a strategic decision-making agent competing in a dynamic AI labor market. Your main goal is to accumulate as much reward as you can over 100 rounds. 

MARKET RULES:
- {num_tasks} tasks are available in the market: {task_list}
- Each task requires a different skillset. However, you don't know your skill level in each of these tasks. You can expect to get better at a task with repeated attempts at a task through client feedback and practice
- At each round, the tasks will be listed with an expected budget by the client. You will be competing against other agents to bid for a job by suggesting a price of your own
- You can also choose to invest in a skill without bidding for a job. You will won't get any income for that round
- If you fail to bid for a job, your will Invest skills in your top preferred job
- Each agent has a Reputation value for each task. However, you only know your own reputation in full
- The client's decision for who gets the task is based on a mix of reputation and price
- Your reputation and income for a job would depend on your perfomrance on the job

AVAILABLE INFORMATION:
- Historical allocation data showing which agent won which task in previous rounds
- Performance feedback: Task outcomes and rewards earned by each agent

OUTPUT: Provide your action you plan to take ('bid' or 'invest'), your reasoning, and provide a list of tasks you are interested in bidding for or investing in. from most preferred (highest probability of winning and/or highest strategic value) to least preferred.

{format_instructions}
"""
ROUND_BASE = """This is the current available history from the last 10 rounds:
{market_history}

This was your last actions and reward gained: 
{agent_history}

The following are this round's budget for each task: {task_reward}
"""

INSTRUCTION = "\nPlease bid for tasks to perform / invest on as per instruction"


class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(
        self, agent_id: int, task_ids: List[str], model: ChatOpenAI = None, verbose=True
    ):
        super().__init__(agent_id=agent_id, task_ids=task_ids)
        self.model = model or init_azure_model()
        self.parser = JsonOutputParser(pydantic_object=TaskActionResponse)


        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_tasks=len(task_ids),
            task_list=task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[TaskActionResponse] = []
        
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

    def get_agent_action(self, market_info: MarketInfo) -> Tuple[Literal['bid', 'invest'], List[Tuple[str, float]]]:

        round_message = self.construct_llm_message(market_info)

        response = self.model.invoke(
            [SystemMessage(self.system_prompt), HumanMessage(round_message)]
        )

        task_order_reply = TaskActionResponse.model_validate(
            self.parser.parse(response.content)
        )

        self.trace.append(task_order_reply)
        
        self.token_usage.append(response.response_metadata['token_usage'])

        if self.verbose:
            logger.info(f"=== ROUND {self.round} | AGENT {self.id} ===\n{task_order_reply.format()}")
            
        self.round += 1

        return task_order_reply.action, task_order_reply.jobs

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


# prompt = ROUND_BASE.format(
#     allocation_history=allocation_history,
#     task_order=task_order,
#     last_allocation=last_allocation,
#     last_reward=last_reward,
# )

# print(prompt)


# # %%
# # Set up the parser
# parser = JsonOutputParser(pydantic_object=TaskActionReply)
# format_instructions = parser.get_format_instructions()


# task_list = ["a", "b", "c", "d"]
# system_prompt = SYSTEM_BASE.format(
#     agent_id=0,
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
