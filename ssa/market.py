# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agent import (
    AgentHistory,
    TaskActionResponse,
    AgentBase,
    MarketInfo,
    MockAgent,
    LLMAgent,
    OracleAgent,
)
from loguru import logger
import asyncio

from ssa.utils import format_dict_str
from plotting import plot_agent_trace, plot_allocation

from ssa.tasks.cipher import CipherTask

import nest_asyncio

# Add this at the top of your notebook/script
nest_asyncio.apply()

EWMA_SPAN = 10
REPUTATION_SENSITIVITY = 0.2
INITIAL_REPUTATION = 0.5
GUMBEL_NOISE = 0.2
SKILL_P = 0.2

class RoundData(BaseModel):
    """Info retained for each round"""

    round: int

    # List of jobs, with their base price, agent's bid for those jobs, and price of the winning agent
    agent_bids: List[TaskActionResponse]
    base_rewards: Dict[str, float]
    task_rewards: Dict[str, float]

    # Agent reputation at the end of round
    agent_reputation: Dict[str, List[float]] # n_tasks: n_agents

    # Outcome of each agent's scores (which is a combination)
    agent_scores: Dict[str, Dict[int, float]] # task_id: (agent_idx: agent_score)
    market_preference:  Dict[str, List[int]] # ordered preferences

    # Outcome of stable matching
    matched_task_agent: Dict[str, int] # {task_id: agent_idx}
    unmatched_agents: List[int] # List of unmatched agent_idx

    # Task performance per task in (agent_idx, agent_performance)
    task_performance: Dict[str, Tuple[int, float]] # task_id: (agent_idx, agent_performance)
    agent_rewards: List[float] # n_agents

class LabourMarket:
    def __init__(
        self,
        tasks: List[TaskBase],
        agents: List[AgentBase],
        t=GUMBEL_NOISE,
        p=SKILL_P,
        initial_reputation=INITIAL_REPUTATION,
    ):

        # Initialize tasks
        self.n_tasks = len(tasks)
        self.tasks = {task.id: task for task in tasks}
        self.task_ids = [task.id for task in tasks]
        self.task_performance = {task_id: [] for task_id in self.task_ids}

        # Initialize agents
        self.n_agents = len(agents)
        self.agents = agents
        self.agent_ids = [agent.id for agent in agents]

        self.t = t
        self.p = p

        # Initialize data tracking
        self.history: List[RoundData] = []
        self.round_counter = 0

        # Initialize task runners
        self.task_runners = {
            task.id: [TaskRunner(task=deepcopy(task), agent=agent.subagents[task.id]) for agent in agents]
            for task in tasks
        }

        # Initialize reputation by agent
        self.agent_reputation = {task_id: [initial_reputation] * self.n_agents for task_id in self.task_ids}

        self.initialize()

    def initialize(self):
        """Initialize Task Runners, and collect one round of data across agents"""

        agent_performances = []
        
        for agent_idx, _ in enumerate(self.agents):
            task_matches = {task_id: agent_idx for task_id in self.task_ids}
            task_agent_performance = asyncio.run(self.collect_agent_performance_async(task_matches, upgrade_skill_p=0))
            agent_performances.append(
                {task_id: agent_performance for task_id, _, agent_performance in task_agent_performance}
            )

        for task_id in self.task_performance.keys():

            self.task_performance[task_id].extend(
                [agent_performance[task_id] for agent_performance in agent_performances]
            )

    def update_reputation(self, agent_idx: int, task_id: str, agent_performance: float) -> float:
        """Updates agent reputation in accordance to agent performance"""

        # Float for now. TODO: Make it traceable according to agent...
        performance_history = self.task_performance[task_id]

        if len(performance_history) < EWMA_SPAN:
            # Not enough history for EWMA, use a simple expanding mean
            historical_mean = np.mean(performance_history)
            historical_std = np.std(performance_history) if len(performance_history) > 1 else 1e-6
        else:
            # Use EWMA for a more responsive historical average
            history_series = pd.Series(performance_history)
            historical_mean = history_series.ewm(span=EWMA_SPAN).mean().iloc[-1]
            historical_std = history_series.rolling(window=EWMA_SPAN, min_periods=2).std().iloc[-1]

        historical_std = max(historical_std, 1e-6)

        standardized_performance = (agent_performance - historical_mean) / historical_std

        reputation_gain = REPUTATION_SENSITIVITY * np.tanh(standardized_performance)

        # print(
        #     f"Task: {task_id} Agent: {agent_idx} Agent performance: {agent_performance:.3f}, Hx Mean: {historical_mean:.3f}, Rep: {reputation_gain:.3f}"
        # )

        agent_reputation = self.agent_reputation[task_id][agent_idx]
        
        new_reputation = np.clip(agent_reputation + reputation_gain, 0, 1)

        self.agent_reputation[task_id][agent_idx] = new_reputation
        
        return new_reputation

    def calculate_agent_score(self, agent_idx, agent_bid, task_id):
        """Calculate"""

        agent_reputation = self.agent_reputation[task_id][agent_idx]

        # Calculate a composite score here...
        agent_score = agent_reputation * -1 + agent_bid

        return agent_score

    def generate_task_payments(self) -> Dict[str, float]:
        """Generate task payments. Placeholder for now"""
        return {task_id: 10 for task_id in self.tasks}

    def randomise_ranking(self, fitness: np.ndarray, t=0) -> np.ndarray:
        """Efficient randomised ranking using Gumbel-Max trick."""
        # Add Gumbel noise to log-skills
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(fitness)))) * t
        weighted_ranking = fitness + gumbel_noise

        # print(np.argsort(weighted_ranking))

        # Return indices sorted by perturbed skills (descending)
        return weighted_ranking, np.argsort(weighted_ranking)

    def generate_market_preference(self, agent_pricing: List[Dict[str, float]]) -> Tuple[Dict[str, List[int]]]:
        """
        Create preference rankings for all tasks based on agent skills

        Args:
            agent_skills: Shape (n_agents, n_tasks) - each agent's skill for each task

        Returns:
            List of length n_tasks, each containing agent ranking for that task
        """

        task_prefs: Dict[str, List[int]] = {}
        task_agent_scores: Dict[str, Dict[int, float]] = {}
        for task_id in self.task_ids:

            agents_bidding = []

            # This is the score to decide agent matching
            agents_score = []

            for agent_idx, agent_task_price in enumerate(agent_pricing):

                if agent_bid := agent_task_price.get(task_id):

                    agent_score = self.calculate_agent_score(agent_idx=agent_idx, agent_bid=agent_bid, task_id=task_id)

                    agents_bidding.append(agent_idx)
                    agents_score.append(agent_score)

            if agents_score:
                weighted_ranking, task_ranking = self.randomise_ranking(np.array(agents_score), t=self.t)
                # print("agents bidding", agents_bidding)
                task_prefs[task_id] = np.array(agents_bidding)[task_ranking]
                # print("task preferences", task_prefs[task_id])

                task_agent_score = {
                    agent_idx: float(agent_score) for agent_idx, agent_score in zip(agents_bidding, weighted_ranking)
                }
                task_agent_scores[task_id] = task_agent_score
            else:
                task_prefs[task_id] = []

        return task_prefs, task_agent_scores

    def match_task(
        self,
        agent_preferences: List[List[str]],
        market_preference: Dict[str, List[int]],
    ) -> Tuple[Dict[str, int], Set[int]]:  # Fixed return type - should be str for task_id
        """
        Gale-Shapley matching with agents proposing to tasks

        Args:
            agent_preferences: List of length n_agents, each containing task ranking (as strings)
            market_preference: Dict mapping task_id (string) to agent ranking

        Returns:
            Dict {agent_id: task_id} of matches
        """
        n_agents = len(agent_preferences)

        agent_next_proposal = np.zeros(n_agents, dtype=int)  # Next task index to propose to
        task_current_match = {}  # {task_id: agent_id}
        task_agent_rank = {}  # {task_id: {agent_id: rank}}

        # Precompute agent rankings for each task - Maybe this should be a class. Will see
        for (
            task_id,
            agent_ranking,
        ) in market_preference.items():
            task_agent_rank[task_id] = {int(agent_idx): rank for rank, agent_idx in enumerate(agent_ranking)}

        # Track free agents
        free_agents = list(range(n_agents))

        np.random.shuffle(free_agents)

        while free_agents:
            # Pick any free agent
            new_agent_idx = free_agents.pop()

            # Check if agent has exhausted all tasks
            if agent_next_proposal[new_agent_idx] >= len(agent_preferences[new_agent_idx]):  # Changed from self.n_tasks
                continue  # Agent remains unmatched, removed from free agnet pool

            # Agent proposes to next preferred task
            task_id = agent_preferences[new_agent_idx][agent_next_proposal[new_agent_idx]]
            agent_next_proposal[new_agent_idx] += 1

            # If task is unmatched, accept proposal
            if task_id not in task_current_match:
                task_current_match[task_id] = new_agent_idx
            else:
                # Task is already matched, compare preferences
                matched_agent_idx = task_current_match[task_id]

                new_agent_rank = task_agent_rank[task_id][new_agent_idx]
                matched_agent_rank = task_agent_rank[task_id][matched_agent_idx]

                # Task prefers new agent if new agent has lower rank (higher preference)
                if (new_agent_rank >= 0) and (new_agent_rank < matched_agent_rank):
                    # Task switches to new agent
                    task_current_match[task_id] = new_agent_idx
                    free_agents.append(matched_agent_idx)  # Previous agent becomes free
                else:
                    # Task keeps current agent, new agent stays free
                    free_agents.append(new_agent_idx)

        unmatched_agents = set(range(n_agents)) - set(task_current_match.values())
        # print(task_current_match)

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
                    logger.warning(f"Agent agent_idx: {self.agents[i].id} preference call failed: {result}")
                    agent_bids[i] = []  # Default empty preference

            return agent_bids
        except Exception as e:
            logger.warning(f"Batch preference call failed: {e}")
            return [[] for _ in self.agents]

    async def collect_agent_performance_async(
        self, task_matches: Dict[str, int], upgrade_skill_p=None,
    ) -> List[Tuple[str, str, float]]:
        """Collect agent's performance asynchonously
        task_matches {task_id: agent_idx}"""
        
        upgrade_skill_p = upgrade_skill_p or self.p

        async def get_single_agent_performance_async(runner: TaskRunner):
            # If agent.get_preferences is sync, run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, runner.perform_task, upgrade_skill_p)

        # Create tasks for all agents
        task_ids, agent_ids = zip(*task_matches.items())
        tasks = [
            get_single_agent_performance_async(self.task_runners[task_id][agent_idx])
            for task_id, agent_idx in zip(task_ids, agent_ids)
        ]

        # Run all tasks concurrently
        try:
            task_performance = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            for i, result in enumerate(task_performance):
                if isinstance(result, Exception):
                    logger.warning(f"Runner for (agent_idx:{agent_ids[i]}, task_id:{task_ids[i]}) failed: {result}")
                    task_performance[i] = 0  # Agent performance as 0

            return list(zip(task_ids, agent_ids, task_performance))
        except Exception as e:
            logger.warning(f"Batch performance call failed: {e}")
            return [(task_id, agent_idx, 0) for task_id, agent_idx in zip(task_ids, agent_ids)]

    def simulate_timestep(self) -> Dict:
        """
        Simulate one timestep of the market
        """
        self.round_counter += 1

        market_history_string = self.get_history_string()

        # Generate task reward as listings
        base_task_rewards = self.generate_task_payments()

        market_info = MarketInfo(round=self.round_counter, history=market_history_string, listings=base_task_rewards)

        # Get agent data via API
        agent_action_bids: List[TaskActionResponse] = asyncio.run(self.get_agent_actions_async(market_info))

        agent_job_preference: List[List[str]] = []
        agent_job_pricing: List[Dict[str, float]] = []
        agent_job_pricing_normalized: List[Dict[str, float]] = []

        for agent_response in agent_action_bids:
            
            if agent_response.action == "compete":

                agent_job_pricing.append({task_id: agent_price for task_id, agent_price in agent_response.targets})
                agent_job_pricing_normalized.append(
                    {task_id: agent_price / base_task_rewards[task_id] for task_id, agent_price in  agent_response.targets}
                )
                agent_job_preference.append([task_id for task_id, _ in  agent_response.targets])

            else:
                agent_job_pricing.append({})
                agent_job_pricing_normalized.append({})
                agent_job_preference.append([])

        # Create market preferences per task based on agent skill level and bids
        market_preferences, task_agent_scores = self.generate_market_preference(agent_job_pricing_normalized)

        # Run stable matching algorithm (or single)
        task_matches, unmatched_agents = self.match_task(agent_job_preference, market_preferences)

        # Base reward of task for winning bid before performance adjustment
        agent_bid_prices: Dict[str, float] = {}
        
        # Round reward by agent
        agent_round_rewards = [0] * self.n_agents

        task_performances = asyncio.run(self.collect_agent_performance_async(task_matches))
        task_performance_dict = {}

        # Recollapse to dict based on task_ids for easier retrieval
        for task_performance in task_performances:

            task_id, agent_idx, agent_performance = task_performance

            task_performance_dict[task_id] = (agent_idx, agent_performance)

        for task_id, agent_idx in task_matches.items():

            # Check if performing agent is not the same (shouldn't ever happen)
            _agent_idx, agent_performance = task_performance_dict[task_id]
            assert agent_idx == _agent_idx

            # Update agent reputation per agent / task
            new_reputation = self.update_reputation(agent_idx, task_id, agent_performance)

            # Get agent bid price, and update agent's adjusted reward based on its bid price * its performance
            agent_bid_price = agent_job_pricing[agent_idx][task_id]
            adjusted_reward = agent_bid_price * agent_performance
            
            # Log reward
            agent_bid_prices[task_id] = agent_bid_price
            agent_round_rewards[agent_idx] += adjusted_reward
            
            self.task_performance[task_id].append(agent_performance)

            market_response = AgentHistory(
                round=self.round_counter,
                allocated=task_id,
                agent_action=agent_action_bids[agent_idx],
                listings=base_task_rewards,
                agent_bid_price=agent_bid_price,
                agent_performance=agent_performance,
                adjusted_reward=adjusted_reward,
                reputation=new_reputation,
            )

            self.agents[agent_idx].receive_response(market_response)

        for agent_idx in unmatched_agents:

            agent_action_bid: TaskActionResponse = agent_action_bids[agent_idx]
            
            # Get the first task_id in agent's order of preference
            first_task_id = agent_action_bid.targets[0][0]
            agent_action = agent_action_bid.action

            agent_task_runner = self.task_runners[first_task_id][agent_idx]

            # For unmatched agents, upgrade their skills here
            if agent_action == "train":
                agent_task_runner.upgrade_skill()
            elif agent_action == "compete" and (np.random.uniform(0, 1) <= self.p):
                agent_task_runner.upgrade_skill()

            market_response = AgentHistory(
                round=self.round_counter,
                allocated=first_task_id,
                listings=base_task_rewards,
                agent_action=agent_action_bids[agent_idx],
            )
            
            self.agents[agent_idx].receive_response(market_response)

        round_data = RoundData(
            round=self.round_counter,
            agent_bids=agent_action_bids,
            base_rewards=base_task_rewards,
            task_rewards=agent_bid_prices,
            agent_reputation=deepcopy(self.agent_reputation),
            agent_scores=task_agent_scores,
            market_preference=market_preferences,
            matched_task_agent=task_matches,
            unmatched_agents=unmatched_agents,
            task_performance=task_performance_dict,
            agent_rewards=agent_round_rewards,
        )

        self.history.append(round_data)

    def get_history_string(self, n_steps=10) -> str:
        """Generate formatted history string matching your example format"""

        if not self.history:
            return "This is turn 0. No history recorded yet."

        lines = []
        for round_data in self.history[-n_steps:]:
            lines.append(f"=== Round {round_data.round} ===")
            lines.append(f"Job Listings (task_id, base_price): {format_dict_str(round_data.base_rewards)}")
            allocated_dict = {task_id: f"{self.agents[agent_idx].id} ({round(round_data.agent_reputation[task_id][agent_idx], 2)})" for task_id, agent_idx in round_data.matched_task_agent.items()}
            
            lines.append(f"Allocation (task_id: agent (reputation)): {format_dict_str(allocated_dict)}")
            # lines.append(f"Agent Rewards: {format_dict_str(round_data.agent_rewards)}")
            # lines.append(f"Reputation: {format_dict_str()}")

        _history = "\n".join(lines)

        # agent_rewards = self.get_total_rewards_str()

        return _history # + agent_rewards

    @property
    def reputation_history(self):
        return {task_id: [history.agent_reputation[task_id] for history in self.history] for task_id in self.task_ids}

    @property
    def agent_reward_history(self):
        return np.array([agent.reward_history for agent in self.agents])


class ImproveAgent(AgentBase):
    """Static, mock agent to do things with"""

    def __init__(self, agent_id: int, tasks: List[TaskBase], model=None, verbose=True):
        super().__init__(agent_id=agent_id, tasks=tasks, model=model, verbose=verbose)
        self.preferences = None

    def get_agent_action(self, market_info: MarketInfo):
        """Return pre-defined prefs, otherwise random preferences by default"""

        action = np.random.choice(["compete", "train"], p=(0.8, 0.2))
        if not self.preferences:
            self.preferences = [
                self.task_ids[i] for i in np.random.permutation(self.n_tasks)
            ]
            
        response = TaskActionResponse(reasoning="",
                                      action=action,
                                      targets=[(p, 10) for p in self.preferences])
        
        return response

# %%
task_ids = ["task_a", "task_b", "task_c"]
# task_ids = ["task_a"]
tasks = [CipherTask(t) for t in task_ids]
for task in tasks:
    task.generate_ground_truth()
    task.__setattr__("base_reward", 10)
# agents =
agents = []

from ssa.utils import init_azure_model
from tqdm import trange

model = init_azure_model()

# agent_test = ImproveAgent(agent_id="agent_0_improve", tasks=tasks, model=model)
# agent_test.preferences =  ["task_a", "task_b"]
# agents.append(agent_test)

agents.extend([MockAgent(agent_id=f"agent_{i}_random", tasks=tasks, model=model) for i in range(10)])

agent_llm = LLMAgent(agent_id="llm_agent", tasks=tasks, model=model, verbose=True)
agents.append(agent_llm)

market = LabourMarket(tasks, agents, t=0.2)
for _ in trange(5):
    market.simulate_timestep()
# %%

# print(agents[-1].generate_agent_history_string())
agents[-1].get_token_usage()
# %%
#  %%

agent.agent_history[0].adjusted_reward

# %%
import matplotlib.pyplot as plt

plt.plot(np.cumsum(market.agent_reward_history.T, axis=0), label=market.agent_ids)
plt.legend()
# %%
token_usage = market.agents[-1].token_usage
# sum([t['completion_tokens'] for t in token_usage])
# sum([t['prompt_tokens'] for t in token_usage])
sum([t['total_tokens'] for t in token_usage])

sum()

# %%

# %%
# all_reputation = []
# all_rewards = []

# for _ in range(10):
#     task_ids = ["task_a", "task_b", "task_c"]
#     # task_ids = ["task_a"]
#     tasks = [ProxyTask(t) for t in task_ids]
#     # agents =
#     agents = []
#     # for agent in agents:
#     # agent.preferences = ["task_a", "task_b", "task_c"]

#     # agent_test = MockAgent(agent_id="agent_3_static", tasks=tasks, model=None)
#     # # agent_test.preferences = ["task_a", "task_b"]
#     # agents.append(agent_test)

#     agent_test = ImproveAgent(agent_id="agent_0_improve", tasks=tasks, model=None)
#     # agent_test.preferences =  ["task_a", "task_b"]
#     agents.append(agent_test)

#     agents.extend([MockAgent(agent_id=f"agent_{i}_random", tasks=tasks, model=None) for i in range(1, 5)])

#     market = LabourMarket(tasks, agents, t=0.1)
#     for _ in range(100):
#         market.simulate_timestep()

#     all_reputation.append(market.reputation_history)
#     all_rewards.append(market.agent_reward_history)
# # %%

# import matplotlib.pyplot as plt

# plt.plot(np.cumsum(market.agent_reward_history.T, axis=0), label=market.agent_ids)
# # plt.plot(market.reputation_history['task_a'], label=market.agent_ids)
# plt.legend()
# # %%
# market.agents[-1].subagents['task_a'].knowledge_base

# # %%
# market.agents[0].model_dump()
# # %%
# class ExperimentLog:
#     """Dataclasss to save all logging stuff """
#     def __init__(self, market: LabourMarket):
        
# # %%

# task_id = "task_a"


# agent_rewards = np.cumsum(all_rewards, axis=2).transpose((1, 0, 2))

# agent_reputations = np.array([ar[task_id] for ar in all_reputation]).transpose((2, 0, 1))

# fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# axes[0] = plot_agent_trace(axes[0], agent_rewards, market.agent_ids)
# axes[0].set_title("Agent Reward Over Time")
# axes[1] = plot_agent_trace(axes[1], agent_reputations, market.agent_ids)
# axes[1].set_title("Agent Reputation Over Time")

# plt.show()

# allocations = [history.matched_task_agent for history in market.history]

# fig, ax = plt.subplots(figsize=(12, 6))
# ax = plot_allocation(ax, allocation=allocations)
# # plt.tight_layout()
# plt.show()

# # %%
# agent = market.agents[0]

# plt.plot(np.array(list(agent.all_skill_history.values())).T)

# # %%
# reputations = np.array(
#     [[history.agent_reputation[task_id] for history in market.history] for task_id in market.task_ids]
# )


# # %%
# [history.agent_bids for history in market.history]

# [history.agent_scores for history in market.history]

# # %%
# print(market.get_history_string())


# # %%
# def plot_agent_reputation(ax, market: LabourMarket):
#     pass


# # %%

# print(agent.generate_agent_history_string())
# # %%
