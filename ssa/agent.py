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
from typing import List, Dict, Optional, Literal, Tuple, Any
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from loguru import logger
from ssa.task import TaskBase, TaskSubAgent, TaskRunner, ProxyAgent

from ssa.tasks.cipher import CipherAgent


class MarketInfo(BaseModel):
    """Data class for market to provide info for agent to action on decisions each round"""
    
    round: int
    history: str
    listings: Dict[str, float] # task_id, budget
    info: Dict[str, Any] = {}


class TaskActionResponse(BaseModel):
    """Data class for agent response"""
    
    reasoning: str = Field(description="Your strategic reasoning for this choice")
    action: Literal['compete', 'train'] = Field(description="Your action for this round. You can either compete for jobs ('compete') or train skills ('train')")
    targets: List[Tuple] = Field(
        description="Your task preferences from highest to lowest priority. For competing: [(task_id_1, price_1), (task_id_2, price_2), ...]. For training: [(task_id, -1)] with only one task."
    )
    
    def format(self):
        if self.action == 'compete':
            target_str = f"Job applications: {self.targets}"
        else:
            task_id = self.targets[0][0] if self.targets else "None"
            target_str = f"Training focus: {task_id}"
            
        return f"\nREASONING: {self.reasoning}\nACTION: {self.action.upper()}\n{target_str}"
    
class AgentHistory(BaseModel):
    """API dataclass for market to return info to each agent per round"""
    
    round: int
    allocated: Optional[str] = None
    agent_action: TaskActionResponse
    listings: Dict[str, float]
    agent_bid_price: Optional[float] = -1
    adjusted_reward: Optional[float] = -1
    agent_performance: Optional[float] = -1
    reputation: Optional[float] = -1

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
        
        self.skill_history = [self.skills]
        self.agent_history: List[AgentHistory] = []
        self.market_history: List[MarketInfo] = []
        self.reputation: Dict[str, Tuple[int, float, float]] = {task_id: (0, 0.5, 0.0) for task_id in self.task_ids} # round, reputation float, delta from previous round
        self.total_reward = 0
        
    @property
    def skills(self) -> List[float]:
        return {task_id: subagent.skill_level for task_id, subagent in self.subagents.items()}

    @abstractmethod
    def get_agent_action(self, market_info: MarketInfo)  -> TaskActionResponse:
        pass

    def receive_response(self, agent_history: AgentHistory):
        self.agent_history.append(agent_history)
        self.total_reward += agent_history.adjusted_reward or 0
        
        allocated_task_id = agent_history.allocated
        
        self.skill_history.append(self.skills)
        
        new_reputation = agent_history.reputation
        
        if allocated_task_id and (new_reputation > 0):
            _, old_reputation, _ = self.reputation[allocated_task_id]
            reputation_delta = new_reputation - old_reputation
            self.reputation[allocated_task_id] = (agent_history.round, new_reputation, reputation_delta)
        
    def get_skill_history(self, task_id: str):
        return np.array([round(hx[task_id], 3) for hx in self.skill_history])
    
    @property
    def all_skill_history(self):
        return {task_id: self.get_skill_history(task_id) for task_id in self.task_ids}
    
    @property
    def reward_history(self):
        return np.array([round(hx.agent_performance, 4) for hx in self.agent_history])
    
    @property
    def allocation_history(self):
            return np.array([round(hx.allocated) for hx in self.agent_history])
        
    def generate_agent_history_string(self, n_steps=10):
        history_lines = []
        
        for round_info in self.agent_history[-n_steps:]:
            action = round_info.agent_action
            round_num = round_info.round
            
            if action.action == "compete":
                if round_info.adjusted_reward >= 0:
                    # Won a job
                    task = round_info.allocated
                    _, new_rep, rep_delta = self.reputation[task]
                    performance = round_info.agent_performance * 10
                    reward = round_info.adjusted_reward
                    
                    history_lines.append(
                        f"R{round_num}: COMPETE {action.targets} → WON {task} "
                        f"(perf: {performance:.1f}/10, reward: ${reward:.2f}, rep: {new_rep - rep_delta:.2f}→{new_rep:.2f})"
                    )
                else:
                    # Lost all bids
                    history_lines.append(f"R{round_num}: COMPETE {action.targets} → NO WINS")
                    
            elif action.action == "train":
                task = round_info.allocated
                history_lines.append(f"R{round_num}: TRAIN → {task}")
        
        # Add current reputation summary
        rep_summary = "Reputation (last known round) - " + " | ".join([
            f"{task}: {rep[1]:.2f} (R{rep[0]})" for task, rep in self.reputation.items()
        ])
        
        return "\n".join(history_lines) + f"\n{rep_summary}"

        
        

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
            
        response = TaskActionResponse(reasoning="",
                                      action="compete",
                                      targets=[(p, 10) for p in self.preferences])
        
        return response


# agent = MockAgent(agent_id="1", task_ids=['a', 'b', 'c'])

# for _ in range(20):
#     preferences = agent.get_preferences()
#     agent.grow_skill(task_id=preferences[0])
#     # agent.grow_skill(task_id="a")

# plot_agent_history(agent)

# plt.plot(agent.get_skill_history("a"))

# # %%

# System prompt

# - Current reputation values for all agents (but not their skills or internal strategies)

SYSTEM_BASE = """You are {agent_id}, a strategic agent competing in an AI labor market simulation over 100 rounds to maximize total reward.

MARKET STRUCTURE:
- {num_tasks} available tasks: {task_list}
- Each round, clients will list tasks with an expected budget as reference. You propose your price to perform these tasks to clients (Can be higher or lower than reference)
- Each task requires different skills; you have hidden skill levels that improve over time
- Your PERFORMANCE on a task is based on your skill level plus randomness
- Your REWARD = performance_ratio * your_offered_price

ACTIONS (choose one per round):
- COMPETE: Submit job applications with your proposed prices
- TRAIN: Skip earning to improve skills in a chosen task

GAME MECHANICS:
- Clients select agents using unknown criteria (likely considering both price and reputation)
- REPUTATION per task reflects your recent performance relative to other agents
- You gain small skill improvements from completing jobs, large improvements from training
- If you fail to win any job you bid for, you automatically train in your top-choice task

AVAILABLE INFORMATION:
- Market history: job listings, allocations, and agent reputations from recent rounds
- Your action history with performance and reputation outcomes

OUTPUT FORMAT:
1. REASONING: Your reasoning for this action
2. ACTION: 'compete' or 'train'
3. TARGETS: If competing, list [(task_id, price), ...] in preference order. If training, specify task_id.
{format_instructions}
"""

ROUND_BASE = """=== ROUND {current_round} ===

RECENT MARKET ACTIVITY (Last {history_window} rounds):
{market_history}

YOUR RECENT PERFORMANCE:
{agent_history}

THIS ROUND'S LISTINGS (task: client_budget): {listings}
"""

INSTRUCTION = "\nChoose to either compete for jobs or train skills based on your strategic analysis."


class LLMAgent(AgentBase):
    """A LLM-based agent to interact with an environment. Has a latent skill vector that is not exposed to the model during LLM calls"""

    def __init__(
        self, agent_id: int, tasks: List[TaskBase], model: ChatOpenAI = None, verbose=True
    ):
        super().__init__(agent_id=agent_id, model=model, tasks=tasks, verbose=verbose)
        self.model = model or init_azure_model()
        self.parser = JsonOutputParser(pydantic_object=TaskActionResponse)


        self.system_prompt = SYSTEM_BASE.format(
            agent_id=self.id,
            num_tasks=len(tasks),
            task_list=self.task_ids,
            format_instructions=self.parser.get_format_instructions(),
        )

        self.verbose = verbose

        self.trace: List[TaskActionResponse] = []
        
        self.token_usage = []
        
        self.round = 0

    def construct_llm_message(self, market_info: MarketInfo):

        return ROUND_BASE.format(
            current_round=market_info.round,
            history_window=10,
            market_history=market_info.history,
            agent_history=self.generate_agent_history_string(),
            listings=market_info.listings,
        ) # + INSTRUCTION

    def rank_skills(self):
        pass

    def get_agent_action(self, market_info: MarketInfo) -> TaskActionResponse:
        
        self.market_history.append(market_info)
        
        round_message = self.construct_llm_message(market_info)
        
        response = self.model.invoke(
            [SystemMessage(self.system_prompt), HumanMessage(round_message)]
        )
        
        if self.verbose: logger.debug(round_message)

        # print(response.content)
        task_order_reply = TaskActionResponse.model_validate(
            self.parser.parse(response.content)
        )
        # print("task reply", task_order_reply)

        self.trace.append((round_message, task_order_reply))
        
        self.token_usage.append(response.response_metadata['token_usage'])

        if self.verbose:
            logger.info(f"=== ROUND {self.round} | AGENT {self.id} ===\n{task_order_reply.format()}")
            
        self.round += 1

        return task_order_reply

    def get_token_usage(self):
        self_token_usage = dict(total_tokens=sum([t['total_tokens'] for t in self.token_usage]),
        completion_tokens=sum([t['completion_tokens'] for t in self.token_usage]),
        prompt_tokens=sum([t['prompt_tokens'] for t in self.token_usage]),)
        
        subagent_token_usage = {} 
        for task_id, subagent in self.subagents.items():
            subagent_token_usage[task_id] = subagent.get_token_usage()

        # Sum all subagent usage
        total_subagent = {}
        for key in ['total_tokens', 'completion_tokens', 'prompt_tokens']:
            total_subagent[key] = sum([usage[key] for usage in subagent_token_usage.values()], 0)
        
        total_token_usage = {
            'total_tokens': self_token_usage['total_tokens'] + total_subagent['total_tokens'],
            'completion_tokens': self_token_usage['completion_tokens'] + total_subagent['completion_tokens'],
            'prompt_tokens': self_token_usage['prompt_tokens'] + total_subagent['prompt_tokens']
        }
        
        return dict(
            total_token_usage=total_token_usage, 
            agent_token_suage=self_token_usage, 
            subagent_token_usage=subagent_token_usage
        )


class OracleAgent(LLMAgent):
    
    def construct_llm_message(self, market_info: MarketInfo):
        
        SKILL_PROMPT = f"\nThis is your current skill: {self.skills}\n"

        return ROUND_BASE.format(
            market_history=market_info.history,
            agent_history=self.generate_agent_history_string(),
            task_reward=market_info.listings,
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

# %%
