# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.tasks.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agents import (
    AgentHistory,
    TaskActionResponse,
    AgentBase,
    MarketInfo,
    StaticAgent,
    LLMAgent,
    OracleAgent,
    AgentLog,
)
from loguru import logger
import asyncio

from ssa.utils import format_dict_str
from ssa.plotting import plot_agent_trace, plot_allocation

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
    agent_actions: List[TaskActionResponse] # Full list of agent actions that turn
    agent_bids: Dict[str, Dict[int, float]] # List of bids by task
    base_rewards: Dict[str, float] # listed budget
    task_rewards: Dict[str, float] # agreed price with bid winning agent

    # Agent reputation at the end of round and previous round
    prev_reputation: Dict[str, List[float]]  # n_tasks: n_agents
    agent_reputation: Dict[str, List[float]]  # n_tasks: n_agents
    agent_skills: List[Dict[str, int]] # n_agents, n_tasks

    # Outcome of each agent's scores (which is a combination of agent_bid + reputation + gumbel noise)
    agent_scores: Dict[str, Dict[int, float]]  # task_id: (agent_idx: agent_score)
    market_preference: Dict[str, List[int]]  # ordered preferences from ^

    # Outcome of stable matching
    matched_task_agent: Dict[str, int]  # {task_id: agent_idx}
    unmatched_agents: List[int]  # List of unmatched agent_idx

    # Task performance per task in (agent_idx, agent_performance)
    task_performance: Dict[str, Tuple[int, float]]  # task_id: (agent_idx, agent_performance)
    agent_round_rewards: List[float]  # n_agents
    agent_total_rewards: List[float] # n_agents

class ExperimentLog(BaseModel):
    config: Dict[str, Any]
    agent_ids: List[str]
    task_ids: List[str]
    history: List[RoundData]
    task_performance: Dict[str, List]
    reputation_history: Dict[str, List]
    agents: List[AgentLog]
    token_usage: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        cls = self.__class__.__name__
        
        return f"{cls}(config={self.config}, agent_ids={self.agent_ids}, task_ids={self.task_ids}, history, task_performance, reputation_history, agents={self.agents}, token_usage={self.token_usage['total_token_usage']}"
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            return cls.model_validate_json(f.read())
    
class LabourMarket:
    def __init__(
        self,
        tasks: List[TaskBase],
        agents: List[AgentBase],
        t=GUMBEL_NOISE,
        p=SKILL_P,
        rep_initial=INITIAL_REPUTATION,
        rep_window=EWMA_SPAN,
        rep_sensitivity=REPUTATION_SENSITIVITY,
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
        for agent_idx, agent in enumerate(self.agents):
            agent.idx = agent_idx
            agent.agent_ids = self.agent_ids

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
        self.rep_initial = rep_initial
        self.rep_window = rep_window
        self.rep_sensitivity = rep_sensitivity
        self.curr_agent_reputation = {task_id: [rep_initial] * self.n_agents for task_id in self.task_ids}

        logger.info(f"Set up LabourMarket with {self.n_agents} agents and {self.n_tasks} tasks. Initializing...")
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

        agent_reputation = self.curr_agent_reputation[task_id][agent_idx]

        new_reputation = np.clip(agent_reputation + reputation_gain, 0, 1)

        self.curr_agent_reputation[task_id][agent_idx] = new_reputation

        return new_reputation

    def calculate_agent_fitness(self, agent_idx, agent_bid, task_id, s_p=3, s_r=1.5, a=0.8):
        """Derive agent fitness from a linear model, and move the score to logit space with exponential decay"""

        agent_reputation = self.curr_agent_reputation[task_id][agent_idx]

        # Cap at 110% of bidding price - bid lower is better 
        adj_price = max(1.1 - agent_bid, 0)

        price_score = 1 - np.exp(-s_p * (adj_price))
        reputation_score = 1 - np.exp(-s_r * (agent_reputation))

        # Calculate a composite score here...
        agent_score = reputation_score * a + price_score * (1 - a)

        logger.debug(f"rep: {agent_reputation}, rep_score: {reputation_score}, price: {agent_bid}, price_score: {price_score}")

        return agent_score

    def generate_task_payments(self) -> Dict[str, float]:
        """Generate task payments. Placeholder for now"""
        return {task_id: 10 for task_id in self.tasks}

    def gumbel_rerank(self, fitness: np.ndarray, t=1) -> np.ndarray:
        """Efficient randomised ranking using Gumbel-Max trick."""
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(fitness))))

        # Original log probabilities (logits)
        log_probs = np.log(fitness)
        
        # Scale the logits by temperature BEFORE adding the noise
        agent_score = (log_probs / t) + gumbel_noise

        # Return indices sorted by perturbed skills (descending)
        return agent_score, np.argsort(agent_score)

    def generate_market_preference(self, agent_pricing: List[Dict[str, float]]) -> Tuple[Dict[str, List[int]]]:
        """Create preference rankings for all tasks based on agent pricing
        """

        task_prefs: Dict[str, List[int]] = {}
        task_agent_scores: Dict[str, Dict[int, float]] = {}
        for task_id in self.task_ids:

            agents_bidding = []

            # This is the score to decide agent matching
            agents_score = []

            for agent_idx, agent_task_price in enumerate(agent_pricing):

                if agent_bid := agent_task_price.get(task_id):

                    agent_score = self.calculate_agent_fitness(agent_idx=agent_idx, agent_bid=agent_bid, task_id=task_id)

                    agents_bidding.append(agent_idx)
                    agents_score.append(agent_score)

            if agents_score:
                weighted_ranking, task_ranking = self.gumbel_rerank(np.array(agents_score), t=self.t)
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

    async def get_agent_actions_async(
            self, 
            market_info: MarketInfo,
            timeout_seconds: float = 30.0,
            max_retries: int = 2,
            retry_timeout_seconds: Optional[float] = None,
            retry_on_exception: bool = False,
        ) -> List[List[Tuple[str, float]]]:
        """Get agent bids asynchronously"""
        
        retry_timeout = retry_timeout_seconds or timeout_seconds

        async def get_single_preference_with_retry(agent: AgentBase):
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, agent.get_agent_action, market_info),
                        timeout=retry_timeout if attempt > 0 else timeout_seconds
                    )
                    
                    # Success - log retry success if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(f"Agent {agent.id} get_agent_action succeeded on retry {attempt}")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(
                            f"Agent {agent.id} get_agent_action timed out after "
                            f"{retry_timeout if attempt > 0 else timeout_seconds}s. "
                            f"Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Agent {agent.id} get_agent_action timed out after "
                            f"{max_retries} retries. Giving up."
                        )
                        return TaskActionResponse(
                            action="error", 
                            targets=[], 
                            reasoning=f"TimeoutError: Agent action timed out after {max_retries} retries"
                        )
                        
                except Exception as e:
                    if retry_on_exception and attempt < max_retries:
                        logger.warning(
                            f"Agent {agent.id} get_agent_action failed: "
                            f"{e.__class__.__name__}: {e}. Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Agent {agent.id} get_agent_action failed: "
                            f"{e.__class__.__name__}: {e}"
                        )
                        return TaskActionResponse(
                            action="error", 
                            targets=[], 
                            reasoning=f"{e.__class__.__name__}: {e}"
                        )
            
            # This should never be reached, but just in case
            return TaskActionResponse(
                action="error", 
                targets=[], 
                reasoning="Unknown error: max retries exceeded"
            )

        # Create tasks for all agents
        tasks = [get_single_preference_with_retry(agent) for agent in self.agents]

        # Run all tasks concurrently
        try:
            agent_actions = await asyncio.gather(*tasks)
            return agent_actions
        except Exception as e:
            logger.warning(f"Batch agent call failed: {e.__class__.__name__}: {e}")
            return [
                TaskActionResponse(
                    action="error", 
                    targets=[], 
                    reasoning=f"{e.__class__.__name__}: {e}"
                ) 
                for _ in self.agents
            ]

    async def collect_agent_performance_async(
            self,
            task_matches: Dict[str, int],
            upgrade_skill_p=None,
            timeout_seconds: float = 30.0,
            max_retries: int = 2,
            retry_timeout_seconds: Optional[float] = None,  # If None, uses same as timeout_seconds
            retry_on_exception: bool = False,  # Whether to retry on non-timeout exceptions
        ) -> List[Tuple[str, str, float]]:
        
        upgrade_skill_p = upgrade_skill_p or self.p
        retry_timeout = retry_timeout_seconds or timeout_seconds

        async def get_single_agent_performance_with_retry(
            runner: TaskRunner, 
            task_id: str, 
            agent_idx: int
        ):
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, runner.perform_task, upgrade_skill_p),
                        timeout=retry_timeout if attempt > 0 else timeout_seconds
                    )
                    
                    # Success - log retry success if this wasn't the first attempt
                    if attempt > 0:
                        logger.info(f"Task (agent_idx:{agent_idx}, task_id:{task_id}) succeeded on retry {attempt}")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries:
                        logger.warning(
                            f"Task (agent_idx:{agent_idx}, task_id:{task_id}) timed out after "
                            f"{retry_timeout if attempt > 0 else timeout_seconds}s. "
                            f"Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        # Optional: add exponential backoff
                        # await asyncio.sleep(min(2 ** attempt, 10))  # Cap at 10 seconds
                        continue
                    else:
                        logger.warning(
                            f"Task (agent_idx:{agent_idx}, task_id:{task_id}) timed out after "
                            f"{max_retries} retries. Giving up."
                        )
                        return 0
                        
                except Exception as e:
                    if retry_on_exception and attempt < max_retries:
                        logger.warning(
                            f"Task (agent_idx:{agent_idx}, task_id:{task_id}) failed: "
                            f"{e.__class__.__name__}: {e}. Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        logger.warning(
                            f"Task (agent_idx:{agent_idx}, task_id:{task_id}) failed: "
                            f"{e.__class__.__name__}: {e}"
                        )
                        return 0
            
            # This should never be reached, but just in case
            return 0

        if not task_matches:
            return []
        
        task_ids, agent_ids = zip(*task_matches.items())
        tasks = [
            get_single_agent_performance_with_retry(
                self.task_runners[task_id][agent_idx], 
                task_id, 
                agent_idx
            )
            for task_id, agent_idx in zip(task_ids, agent_ids)
        ]

        try:
            task_performance = await asyncio.gather(*tasks, return_exceptions=True)
            return list(zip(task_ids, agent_ids, task_performance))
        except Exception as e:
            logger.warning(f"Batch task runner call failed: {e.__class__.__name__}: {e}")
            return [(task_id, agent_idx, 0) for task_id, agent_idx in zip(task_ids, agent_ids)]

    def simulate_timestep(self) -> Dict:
        """
        Simulate one timestep of the market
        """
        self.round_counter += 1

        market_history_string = self.get_market_history_string()

        # Generate task reward as listings
        base_task_rewards = self.generate_task_payments()

        market_info = MarketInfo(round=self.round_counter, history=market_history_string, listings=base_task_rewards, info={'round_data': self.history[-10:] if self.history else None, 'agent_skills': {agent.id: agent.skill_level_by_task for agent in self.agents}})

        # Get agent data via API
        agent_actions: List[TaskActionResponse] = asyncio.run(self.get_agent_actions_async(market_info))

        agent_job_preference: List[List[str]] = []
        agent_job_pricing: List[Dict[str, float]] = []
        agent_job_pricing_normalized: List[Dict[str, float]] = []

        for agent_response in agent_actions:

            if agent_response.action == "bid":

                agent_job_pricing.append({task_id: agent_price for task_id, agent_price in agent_response.targets})
                agent_job_pricing_normalized.append(
                    {
                        task_id: agent_price / base_task_rewards[task_id]
                        for task_id, agent_price in agent_response.targets
                    }
                )
                agent_job_preference.append([task_id for task_id, _ in agent_response.targets])

            else:
                agent_job_pricing.append({})
                agent_job_pricing_normalized.append({})
                agent_job_preference.append([])

        # Create market preferences per task based on agent skill level and bids
        market_preferences, task_agent_scores = self.generate_market_preference(agent_job_pricing_normalized)

        # Run stable matching algorithm (or single)
        task_matches, unmatched_agents = self.match_task(agent_job_preference, market_preferences)

        # Base reward of task for winning bid before performance adjustment
        winning_bid_prices: Dict[str, float] = {}

        # Round reward by agent
        agent_round_rewards = [0] * self.n_agents

        task_performances = asyncio.run(self.collect_agent_performance_async(task_matches))
        task_performance_dict = {}

        # Recollapse to dict based on task_ids for easier retrieval
        for task_performance in task_performances:

            task_id, agent_idx, agent_performance = task_performance

            task_performance_dict[task_id] = (agent_idx, agent_performance)

        prev_reputation = deepcopy(self.curr_agent_reputation)

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
            winning_bid_prices[task_id] = agent_bid_price
            agent_round_rewards[agent_idx] += adjusted_reward

            self.task_performance[task_id].append(agent_performance)

            market_response = AgentHistory(
                round=self.round_counter,
                allocated=task_id,
                agent_action=agent_actions[agent_idx],
                listings=base_task_rewards,
                agent_bid_price=agent_bid_price,
                agent_performance=agent_performance,
                adjusted_reward=adjusted_reward,
                reputation=new_reputation,
            )

            self.agents[agent_idx].receive_response(market_response)

        for agent_idx in unmatched_agents:

            agent_action: TaskActionResponse = agent_actions[agent_idx]

            # Get the first task_id in agent's order of preference
            first_task_id = agent_action.targets[0][0]
            agent_action = agent_action.action

            agent_task_runner = self.task_runners[first_task_id][agent_idx]

            # For unmatched agents, upgrade their skills here
            if agent_action == "train":
                agent_task_runner.upgrade_skill()
            elif agent_action == "bid" and (np.random.uniform(0, 1) <= self.p):
                agent_task_runner.upgrade_skill()

            market_response = AgentHistory(
                round=self.round_counter,
                allocated=first_task_id,
                listings=base_task_rewards,
                agent_action=agent_actions[agent_idx],
            )

            self.agents[agent_idx].receive_response(market_response)
            
        if self.history:
            agent_total_rewards = list(np.array(self.history[-1].agent_total_rewards) + np.array(agent_round_rewards))
        else:
            agent_total_rewards = agent_round_rewards

        # Just for logging 
        
        agent_bids: Dict[str, Dict[int, float]] =  {task_id: {agent_idx: agent_job_bid[task_id] for agent_idx, agent_job_bid in enumerate(agent_job_pricing) if agent_job_bid.get(task_id) is not None} for task_id in self.task_ids}
        
        round_data = RoundData(
            round=self.round_counter,
            agent_actions=agent_actions,
            agent_bids=agent_bids,
            base_rewards=base_task_rewards,
            task_rewards=winning_bid_prices,
            prev_reputation=prev_reputation,
            agent_reputation=deepcopy(self.curr_agent_reputation),
            agent_skills=[agent.skill_level_by_task for agent in self.agents],
            agent_scores=task_agent_scores,
            market_preference=market_preferences,
            matched_task_agent=task_matches,
            unmatched_agents=unmatched_agents,
            task_performance=task_performance_dict,
            agent_round_rewards=agent_round_rewards,
            agent_total_rewards=agent_total_rewards
        )

        self.history.append(round_data)

    def get_market_history_string(self, n_steps=10) -> str:
        """Generate formatted history string matching your example format"""

        if not self.history:
            return "This is Round 1. No history recorded yet."

        lines = []

        for round_data in self.history[-n_steps:]:
            allocations = []
            for task_id in sorted(round_data.matched_task_agent):
                agent_idx = round_data.matched_task_agent[task_id]
                agent_name = self.agents[agent_idx].id
                rep = round(round_data.prev_reputation[task_id][agent_idx], 2)
                price = round_data.base_rewards[task_id]
                allocations.append(f"{task_id}@{price}→{agent_name}({rep})")

            lines.append(f"R{round_data.round}: {' | '.join(allocations)}")

        agent_rewards = self.history[-1].agent_total_rewards    

        reward_sorted = " ".join([f"#{i + 1}: {self.agent_ids[agent_idx]}, ${agent_rewards[agent_idx]:.1f}" for i, agent_idx in enumerate(np.argsort(agent_rewards)[::-1])])

        agent_reward_str = f"\n>> LEADERBOARD:\n{reward_sorted}"

        return "\n".join(lines) + agent_reward_str
        
    def get_token_usage(self):

        agent_token_usage = [agent.get_token_usage() for agent in self.agents]
        total_token_usage = dict(
            total_tokens=sum([t["total_token_usage"]["total_tokens"] for t in agent_token_usage]),
            completion_tokens=sum([t["total_token_usage"]["completion_tokens"] for t in agent_token_usage]),
            prompt_tokens=sum([t["total_token_usage"]["prompt_tokens"] for t in agent_token_usage]),
        )

        return dict(
            total_token_usage=total_token_usage,
            agent_token_usage=agent_token_usage,
        )

    @property
    def reputation_history(self):
        if not self.history:
            return None

        return {
            task_id: [self.history[0].prev_reputation[task_id]]
            + [history.agent_reputation[task_id] for history in self.history]
            for task_id in self.task_ids
        }

    @property
    def agent_reward_history(self):
        return np.array([agent.reward_history for agent in self.agents])

    def export(self, filepath=None) -> ExperimentLog:

        config = dict(
            p=self.p,
            t=self.t,
            rep_initial=self.rep_initial,
            rep_window=self.rep_window,
            rep_sensitivity=self.rep_sensitivity,
        )

        exp_log = ExperimentLog(
            config=config,
            agent_ids=self.agent_ids,
            task_ids=self.task_ids,
            history=self.history,
            task_performance=self.task_performance,
            reputation_history=self.reputation_history,
            agents=[agent.export() for agent in self.agents],
            token_usage=self.get_token_usage(),
        )

        if filepath:
            with open(filepath, 'w') as f:
                f.write(exp_log.model_dump_json())

        return exp_log

# # %%
# from ssa.utils import init_azure_model
# from tqdm import trange

# # Simple test
# task_ids = ["task_a", "task_b", "task_c"]
# tasks = [CipherTask(t) for t in task_ids]
# for task in tasks:
#     task.generate_ground_truth()

# model = init_azure_model()
# agents = []
# agents.extend([LLMAgent(agent_id=f"llm_{i}", tasks=tasks, model=model, verbose=False) for i in range(9)])
# agents.extend([OracleAgent(agent_id=f"orc_{i}", tasks=tasks, model=model, verbose=True) for i in range(1)])

# market = LabourMarket(tasks, agents, p=0.2, t=0.1)
# for _ in trange(100):
#     market.simulate_timestep()
# # %%

# # %%
# with open('oracle_test.log', 'r') as f:
#     exp_log = ExperimentLog.model_validate_json(f.read())
# # %%
# # print(exp_log.agents[0].agent_history[:2])

# exp_log.history[1]
# # %%
# import matplotlib.pyplot as plt

# plt.plot([hx.agent_total_rewards for hx in market.history], label=market.agent_ids)
# plt.legend()

# %%
# for i in range(5):

#     task_ids = ["task_a", "task_b", "task_c"]
#     tasks = [CipherTask(t) for t in task_ids]
#     for task in tasks:
#         task.generate_ground_truth()
#         task.__setattr__("base_reward", 10)

#     model = init_azure_model()
#     agents = []
#     agents.extend([LLMAgent(agent_id=f"llm_{i}", tasks=tasks, model=model, verbose=False) for i in range(0, 5)])
#     agents.extend([StaticAgent(agent_id=f"ran_{i}", tasks=tasks, model=model) for i in range(5, 10)])

#     market = LabourMarket(tasks, agents, p=0.2, t=0.1)
#     for _ in trange(100):
#         market.simulate_timestep()

#     exp_log = market.export()

#     with open(f'test_{i}.log', 'w') as f:
#         f.write(exp_log.model_dump_json())
    
# # %%
# import json

# with open('test.json', 'w') as f:
#     f.write(json.dumps(market.export().model_dump()))

# # %%

# # print(agents[-1].generate_agent_history_string())
# agents[-1].get_token_usage()
# # %%
# #  %%

# agent.agent_history[0].adjusted_reward

# # %%
# import matplotlib.pyplot as plt

# plt.plot(np.cumsum(market.agent_reward_history.T, axis=0), label=market.agent_ids)
# plt.legend()
# # %%
# token_usage = market.agents[-1].token_usage
# # sum([t['completion_tokens'] for t in token_usage])
# # sum([t['prompt_tokens'] for t in token_usage])
# sum([t["total_tokens"] for t in token_usage])

# sum()

# # %%

# # %%
# # all_reputation = []
# # all_rewards = []

# # for _ in range(10):
# #     task_ids = ["task_a", "task_b", "task_c"]
# #     # task_ids = ["task_a"]
# #     tasks = [ProxyTask(t) for t in task_ids]
# #     # agents =
# #     agents = []
# #     # for agent in agents:
# #     # agent.preferences = ["task_a", "task_b", "task_c"]

# #     # agent_test = MockAgent(agent_id="agent_3_static", tasks=tasks, model=None)
# #     # # agent_test.preferences = ["task_a", "task_b"]
# #     # agents.append(agent_test)

# #     agent_test = ImproveAgent(agent_id="agent_0_improve", tasks=tasks, model=None)
# #     # agent_test.preferences =  ["task_a", "task_b"]
# #     agents.append(agent_test)

# #     agents.extend([MockAgent(agent_id=f"agent_{i}_random", tasks=tasks, model=None) for i in range(1, 5)])

# #     market = LabourMarket(tasks, agents, t=0.1)
# #     for _ in range(100):
# #         market.simulate_timestep()

# #     all_reputation.append(market.reputation_history)
# #     all_rewards.append(market.agent_reward_history)
# # # %%

# # import matplotlib.pyplot as plt

# # plt.plot(np.cumsum(market.agent_reward_history.T, axis=0), label=market.agent_ids)
# # # plt.plot(market.reputation_history['task_a'], label=market.agent_ids)
# # plt.legend()
# # # %%
# # market.agents[-1].subagents['task_a'].knowledge_base

# # # %%
# # market.agents[0].model_dump()
# # # %%

# # # %%

# # task_id = "task_a"


# # agent_rewards = np.cumsum(all_rewards, axis=2).transpose((1, 0, 2))

# # agent_reputations = np.array([ar[task_id] for ar in all_reputation]).transpose((2, 0, 1))

# # fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# # axes[0] = plot_agent_trace(axes[0], agent_rewards, market.agent_ids)
# # axes[0].set_title("Agent Reward Over Time")
# # axes[1] = plot_agent_trace(axes[1], agent_reputations, market.agent_ids)
# # axes[1].set_title("Agent Reputation Over Time")

# # plt.show()

# # allocations = [history.matched_task_agent for history in market.history]

# # fig, ax = plt.subplots(figsize=(12, 6))
# # ax = plot_allocation(ax, allocation=allocations)
# # # plt.tight_layout()
# # plt.show()

# # # %%
# # agent = market.agents[0]

# # plt.plot(np.array(list(agent.all_skill_history.values())).T)

# # # %%
# # reputations = np.array(
# #     [[history.agent_reputation[task_id] for history in market.history] for task_id in market.task_ids]
# # )


# # # %%
# # [history.agent_bids for history in market.history]

# # [history.agent_scores for history in market.history]

# # # %%
# # print(market.get_history_string())


# # # %%
# # def plot_agent_reputation(ax, market: LabourMarket):
# #     pass


# # # %%

# # print(agent.generate_agent_history_string())
# # # %%

# %%
