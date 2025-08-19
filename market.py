# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
from pydantic import BaseModel
from task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from agent import MarketResponse, AgentBase, MarketInfo, MockAgent, LLMAgent, OracleAgent
from loguru import logger
import asyncio

from utils import format_dict_str

import nest_asyncio

# Add this at the top of your notebook/script
nest_asyncio.apply()

class RoundData(BaseModel):
    """Info retained for each round"""

    round: int
    agent_preference: Any
    market_preference: Any
    matched_task_agent: Any
    unmatched_agents: Any
    base_reward: Any
    adjusted_reward: Any


class LabourMarket:
    def __init__(self, tasks: List[TaskBase], agents: List[AgentBase]):
        self.n_tasks = len(tasks)
        self.tasks = {task.id: task for task in tasks}
        self.task_ids = [task.id for task in tasks]

        self.n_agents = len(agents)
        self.agents = agents
        self.round_history: List[RoundData] = []
        self.round_counter = 0

        self.task_runners = {
            task.id: [TaskRunner(task = deepcopy(task), agent = agent.subagents[task.id]) for agent in agents] for task in tasks
        }

    def generate_tasks(self) -> np.ndarray:
        """Generate task payments. Placeholder for now"""
        return {task_id: 10 for task_id in self.tasks}

    def randomise_ranking(self, fitness: np.ndarray, t=0) -> np.ndarray:
        """Efficient randomised ranking using Gumbel-Max trick."""
        # Add Gumbel noise to log-skills
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(fitness)))) * t
        weighted_ranking = fitness + gumbel_noise

        # Return indices sorted by perturbed skills (descending)
        return np.argsort(-weighted_ranking)

    def generate_market_preference(self, agent_pricing: List[Dict[str, float]]) -> List[np.ndarray]:
        """
        Create preference rankings for all tasks based on agent skills

        Args:
            agent_skills: Shape (n_agents, n_tasks) - each agent's skill for each task

        Returns:
            List of length n_tasks, each containing agent ranking for that task
        """

        task_prefs = {}
        for task_id in self.task_ids:
            
            agents_bidding = []
            
            # This is the score 
            agents_score = []
            
            for agent_idx, agent_task_price in enumerate(agent_pricing):
                
                if agent_price := agent_task_price.get(task_id):
                    agent_reputation = self.agents[agent_idx].reputation[task_id]
                    agent_score = agent_reputation * -1 + agent_price
                    
                    agents_bidding.append(agent_idx)
                    agents_score.append(agent_score)
            
            task_ranking = self.randomise_ranking(agents_score)

            task_prefs[task_id] = np.array(agents_bidding)[task_ranking]

        return task_prefs

    def match_task(
        self,
        agent_preferences: List[List[str]],
        market_preference: Dict[str, List[int]],
    ) -> Tuple[
        Dict[str, int], Set[int]
    ]:  # Fixed return type - should be str for task_id
        """
        Gale-Shapley matching with agents proposing to tasks

        Args:
            agent_preferences: List of length n_agents, each containing task ranking (as strings)
            market_preference: Dict mapping task_id (string) to agent ranking

        Returns:
            Dict {agent_id: task_id} of matches
        """
        n_agents = len(agent_preferences)

        # 
        agent_next_proposal = np.zeros(
            n_agents, dtype=int
        )  # Next task index to propose to
        task_current_match = {}  # {task_id: agent_id}
        task_agent_rank = {}  # {task_id: {agent_id: rank}}

        # Precompute agent rankings for each task - Maybe this should be a class. Will see
        for (
            task_id,
            agent_ranking,
        ) in market_preference.items():  # Changed from enumerate
            task_agent_rank[task_id] = {
                agent_id: rank for rank, agent_id in enumerate(agent_ranking)
            }

        # Track free agents
        free_agents = set(range(n_agents))

        while free_agents:
            # Pick any free agent
            agent = free_agents.pop()

            # Check if agent has exhausted all tasks
            if agent_next_proposal[agent] >= len(
                agent_preferences[agent]
            ):  # Changed from self.n_tasks
                continue  # Agent remains unmatched

            # Agent proposes to next preferred task
            task_id = agent_preferences[agent][
                agent_next_proposal[agent]
            ]  # Now returns string
            agent_next_proposal[agent] += 1

            # If task is unmatched, accept proposal
            if task_id not in task_current_match:
                task_current_match[task_id] = agent
            else:
                # Task is already matched, compare preferences
                current_agent = task_current_match[task_id]

                # Task prefers new agent if new agent has lower rank (higher preference)
                if (
                    task_agent_rank[task_id].get(agent, -1)
                    > task_agent_rank[task_id].get(agent, -1)
                ):
                    # Task switches to new agent
                    task_current_match[task_id] = agent
                    free_agents.add(current_agent)  # Previous agent becomes free
                else:
                    # Task keeps current agent, new agent stays free
                    free_agents.add(agent)

        unmatched_agents = set(range(n_agents)) - set(task_current_match.values())
        
        # Return matched, unmatched agents from agent perspective
        return task_current_match, unmatched_agents
    
    def get_total_rewards(self) -> List[float]:
        
        return [a.total_reward for a in self.agents]
    
    def get_total_rewards_str(self) -> str:
        total_rewards_str = """\n\nCumulative Rewards: ["""
        
        total_rewards_str += ", ".join(f"{agent.id}: {agent.total_reward:.2f}" for agent in self.agents)
        total_rewards_str += "]"
        
        return total_rewards_str
        
    async def get_agent_actions_async(self, market_info: MarketInfo) -> List[List[Tuple[str, float]]]:
        """Get agent bids asynchronously"""
        
        async def get_single_preference_async(agent: AgentBase):
            # If agent.get_preferences is sync, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, agent.get_agent_action, market_info)
        
        # Create tasks for all agents
        tasks = [get_single_preference_async(agent) for agent in self.agents]
        
        # Run all tasks concurrently
        try:
            agent_bids = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for i, result in enumerate(agent_bids):
                if isinstance(result, Exception):
                    logger.warning(f"Agent {self.agents[i].id} preference call failed: {result}")
                    agent_bids[i] = []  # Default empty preference
                    
            return agent_bids
        except Exception as e:
            logger.warning(f"Batch preference call failed: {e}")
            return [[] for _ in self.agents]     

    def simulate_timestep(self) -> Dict:
        """
        Simulate one timestep of the market
        """
        # Dummy - Generate tasks and payments.
        # max_payments = self.generate_tasks()

        self.round_counter += 1
        
        market_history_string = self.get_history_string()
        
        task_base_rewards = format_dict_str({task_id: task.base_reward for task_id, task in self.tasks.items()})
        
        market_info = MarketInfo(history=market_history_string, task_reward={})
        
        # Get agent data via API
        agent_action_bids = asyncio.run(self.get_agent_actions_async(market_info))
        
        agent_preference = [[task_id for task_id, _ in agent_bid] for _, agent_bid in agent_action_bids]
        agent_job_pricing: List[Dict[str, float]] = [{task_id: agent_price for task_id, agent_price in agent_bid} for agent_action, agent_bid in agent_action_bids if agent_action == "bid"]
        
        print(agent_preference)
        
        # Create task preferences based on agent skill level and bids
        market_preferences = self.generate_market_preference(agent_job_pricing)

        # Run stable matching algorithm
        task_matches, unmatched_agents = self.match_task(
            agent_preference, market_preferences
        )

        base_reward_dict = {}
        agent_reward_dict = {}

        matched_task_agent = {}
        for task_id, agent_idx in task_matches.items():
            agent_id = self.agents[agent_idx].id
            matched_task_agent[task_id] = agent_id

            # This step is where the agent actually does the task
            agent_performance = self.task_runners[task_id][
                agent_idx
            ].perform_task()
            
            base_reward = agent_job_pricing[agent_idx][task_id]
            adjusted_reward = round(base_reward * agent_performance, 3)
            
            # Update agent reputation here
            
            
            # Log reward
            base_reward_dict[task_id] = base_reward
            agent_reward_dict[agent_id] = round(adjusted_reward, 3)

            market_response = MarketResponse(
                round=self.round_counter,
                allocated=task_id,
                preference=agent_preference[agent_idx],
                task_id=task_id,
                base_reward=base_reward,
                adjusted_reward=adjusted_reward,
            )
            
            self.agents[agent_idx].receive_response(market_response)

        for agent_idx in unmatched_agents:
            
            # Get the first item in order, and get the task_id in first item
            task_id = agent_preference[agent_idx][0]
            
            # For unmatched agents, upgrade their skills here
            agent_performance = self.task_runners[task_id][
                agent_idx
            ].upgrade_skill()
            
            market_response = MarketResponse(round=self.round_counter, preference=agent_preference[agent_idx])
            self.agents[agent_idx].receive_response(market_response)
            
            agent_reward_dict[self.agents[agent_idx].id] = 0 
            
        round_data = RoundData(
            round=self.round_counter,
            agent_preference=agent_action_bids,
            market_preference=market_preferences,
            matched_task_agent=matched_task_agent,
            unmatched_agents=unmatched_agents,
            base_reward=base_reward_dict,
            adjusted_reward=agent_reward_dict,
        )

        self.round_history.append(round_data)

    def get_history_string(self, n_steps=10) -> str:
        """Generate formatted history string matching your example format"""

        if not self.round_history:
            return "This is turn 0. No history recorded yet."

        lines = []
        for round_data in self.round_history[-n_steps:]:
            lines.append(f"=== Round {round_data.round} ===")
            lines.append(f"Maximum Task Rewards: {format_dict_str(round_data.base_reward)}")
            lines.append(f"Allocation: {format_dict_str(round_data.matched_task_agent)}")
            lines.append(f"Agent Rewards: {format_dict_str(round_data.adjusted_reward)}")

        _history = "\n".join(lines)
        
        agent_rewards = self.get_total_rewards_str()
        
        return _history + agent_rewards

task_ids = ["task_a", "task_b", "task_c", "task_d"]
tasks = [ProxyTask(t) for t in task_ids]
agents = [MockAgent(agent_id='abc', tasks=tasks, model=None) for _ in range(5)]

market = LabourMarket(tasks, agents)
market.simulate_timestep()

###  OLD EXPERIMENTS
# # %%
# # Quick test # 1

# # Create market and agents
# task_ids = ["task_a", "task_b", "task_c", "task_d"]
# tasks = [MockTask(t) for t in task_ids]
# agents = [MockAgent(f"agent_{i}", task_ids=task_ids) for i in range(7)]
# agents.append(LLMAgent(agent_id="llm_agent", task_ids=task_ids))

# market = LabourMarket(tasks=tasks, agents=agents)

# # Run simulation

# for _ in range(20):
#     market.simulate_timestep()
# # market.simulate_timestep()
    
# # %%
# import matplotlib.pyplot as plt
# agent = market.agents[-1]

# plt.figure()
# for task_id, skill_hx in agent.all_skill_history.items():
#     plt.plot(skill_hx, label=task_id)     
# plt.legend()
# plt.show()

# plt.figure()
# for agent in agents: 
#     plt.plot(np.cumsum(agent.reward_history), label=agent.id)

# plt.legend()
# plt.show()


# # %% 
# # Create market and agents
# task_ids = ["task_a", "task_b", "task_c", "task_d"]
# tasks = [MockTask(t) for t in task_ids]
# agents = [MockAgent(f"agent_{i}", task_ids=task_ids) for i in range(9)]

# for ix in [0, 1, 2]:
#     agents[ix].preferences = ["task_a", "task_b", "task_c", "task_d"]

# for ix in [3, 4, 5]:
#     agents[ix].preferences = ["task_b", "task_c", "task_a", "task_d"]

# for ix in [6, 7, 8]:
#     agents[ix].preferences = ["task_c", "task_a", "task_b", "task_d"]
    
# agents.append(LLMAgent(agent_id="llm_agent", task_ids=task_ids))

# market = LabourMarket(tasks=tasks, agents=agents)

# # Run simulation
# for _ in range(20):
#     market.simulate_timestep()
# # market.simulate_timestep()

# # %% 
# # LLM vs Oracle
# # Create market and agents
# task_ids = ["task_a", "task_b", "task_c"]
# tasks = [MockTask(t) for t in task_ids]
# agents = [LLMAgent(f"llm_agent_{i}", task_ids=task_ids, verbose=False) for i in range(4)]
# agents.append(OracleAgent(agent_id="oracle_agent", task_ids=task_ids))

# market = LabourMarket(tasks=tasks, agents=agents)

# # Run simulation
# for _ in range(20):
#     market.simulate_timestep()


# # %%
# import matplotlib.pyplot as plt
# agent = market.agents[-1]

# plt.figure()
# for task_id, skill_hx in agent.all_skill_history.items():
#     plt.plot(skill_hx, label=task_id)     
# plt.legend()
# plt.show()

# plt.figure()
# for agent in agents: 
#     plt.plot(np.cumsum(agent.reward_history), label=agent.id)

# plt.legend()
# plt.show()

# # %%
# agent: LLMAgent = market.agents[4]

# plt.figure()
# for task_id, skill_hx in agent.all_skill_history.items():
#     plt.plot(skill_hx, label=task_id)     
# plt.legend()
# plt.show()

# # %%
# np.sum(agent.total_tokens for agent in agents)
# # print(market.get_history_string())

# # agent = agents[0]
# # print(agent.generate_agent_history_string())

# # # agent_preferences = [agent.get_preferences() for agent in agents]
# # agent_preferences

# # agent_skills = np.array(
# #     [agent.get_skills() for agent in agents]
# # )
# # print(agent_preferences)
# # market_preference = {
# #     0: [0, 1, 2, 3, 4],
# #     1: [3, 0, 1, 2, 4],
# #     2: [2, 3, 4, 1, 0],
# # }

# # market.match_task(agent_preferences, market_preference)

# # %%

# %%
