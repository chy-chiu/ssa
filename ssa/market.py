# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.tasks.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agents import (
    AgentBase,
    StaticAgent,
    LLMAgent,
    OracleAgent,
    AgentLog,
)
from loguru import logger
import asyncio

from ssa.utils import format_dict_str
from ssa.plotting import plot_agent_trace, plot_allocation
from ssa.common import Job, ExperimentLog, AgentLog, RoundData, JobHistory, AgentHistory, AgentPerformance, AgentActionResponse, MarketInfo

from ssa.tasks.cipher import CipherTask

import nest_asyncio

# Add this at the top of your notebook/script
nest_asyncio.apply()

EWMA_SPAN = 5
REPUTATION_PRIOR_STRENGTH = 1
INITIAL_REPUTATION = 0.5
GUMBEL_NOISE = 0.2
SKILL_P = 0.2
L = 0.85

class LabourMarket:
    def __init__(
        self,
        jobs: List[Job],
        tasks: List[TaskBase],
        agents: List[AgentBase],
        t=GUMBEL_NOISE,
        p=SKILL_P,
        rep_initial=INITIAL_REPUTATION,
        rep_window=EWMA_SPAN,
        rep_sensitivity=REPUTATION_PRIOR_STRENGTH,
        rep_lambda=L,
    ):

        # Initialize tasks
        self.n_tasks = len(tasks)
        self.tasks = {task.id: task for task in tasks}
        self.task_ids = [task.id for task in tasks]

        # Initialize jobs
        self.jobs = {job.id: job for job in jobs}
        self.job_ids = [job.id for job in jobs]
        self.n_jobs = len(jobs)

        self.job_performance: List[AgentPerformance] = []
        self.job_to_task_id = {job.id: job.task_id for job in jobs}

        # Initialize agents
        self.n_agents = len(agents)
        self.agents = agents
        self.agent_ids = [agent.id for agent in agents]

        # Inject other agent info to agent class
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

        # Initialize task reputation by agent
        self.rep_initial = rep_initial
        self.rep_window = rep_window
        self.rep_sensitivity = rep_sensitivity
        self.rep_lambda = rep_lambda
        self.curr_agent_reputation = {task_id: [self.rep_initial] * self.n_agents for task_id in self.task_ids}

        logger.info(f"Set up LabourMarket with {self.n_agents} agents and {self.n_tasks} tasks. Initializing...")
        self.initialize()

    def get_job_performance(self, agent_idx=None, task_id=None, job_id=None, filter_initial=True):
        return [
            perf
            for perf in self.job_performance
            if (agent_idx is None or perf.agent_idx == agent_idx)
            and (task_id is None or perf.task_id == task_id)
            and (job_id is None or perf.job_id == job_id)
            and (not filter_initial or perf.round >= 0)
        ]

    def initialize(self):
        """Initialize Task Runners, and collect one round of data across agents"""

        agent_performances = []

        # Get all agent to do each job once
        for agent_idx, _ in enumerate(self.agents):
            job_matches = {job_id: agent_idx for job_id in self.job_ids}
            job_agent_performance = asyncio.run(self.execute_jobs_async(job_matches, upgrade_skill_p=0))

            _job_agent_performance = [
                AgentPerformance(
                    agent_idx=agent_idx,
                    agent_id=self.agent_ids[agent_idx],
                    round=-1,
                    task_id=self.job_to_task_id[job_id],
                    job_id=job_id,
                    performance=performance,
                )
                for job_id, (agent_idx, performance) in job_agent_performance.items()
            ]

            self.job_performance.extend(_job_agent_performance)

            agent_performances.append(
                {job_id: performance for job_id, (_, performance) in job_agent_performance.items()}
            )

        for job_id in self.job_ids:

            for agent_idx, _ in enumerate(self.agents):
                agent_performance = agent_performances[agent_idx][job_id]
                task_id = self.job_to_task_id[job_id]
                initial_rep = self.update_reputation(agent_idx, task_id, agent_performance)
                self.agents[agent_idx].reputation[task_id] = (0, initial_rep, 0)

    def update_reputation(self, agent_idx: int, task_id: str, agent_performance: float) -> float:
        """Updates agent reputation for a specific task"""

        community_performance_history = np.array([v.performance for v in self.get_job_performance(task_id=task_id)])
        agent_performance_history = np.array(
            [v.performance for v in self.get_job_performance(task_id=task_id, agent_idx=agent_idx)]
        )

        community_baseline_performance = (
            np.mean(community_performance_history[-self.rep_window :])
            if len(community_performance_history) > 0
            else self.rep_initial
        )

        # accumulate agent evidence r,s with forgetting (Eq. 12/13 style recursion)
        if agent_performance_history is not None and len(agent_performance_history) > 0:
            r = np.sum(
                agent_performance_history
                * np.array([self.rep_lambda**i for i, _ in enumerate(agent_performance_history)][::-1])
            )
            s = np.sum(
                (1 - agent_performance_history)
                * np.array([self.rep_lambda**i for i, _ in enumerate(agent_performance_history)][::-1])
            )
        else:
            r = 0.0
            s = 0.0

        # incorporate the new performance (one more recursive step)
        r = self.rep_lambda * r + agent_performance
        s = self.rep_lambda * s + (1.0 - agent_performance)

        # expectation with base-rate prior (subjective-logic form)
        rep = (r + self.rep_sensitivity * community_baseline_performance) / (r + s + self.rep_sensitivity)

        new_reputation = np.clip(rep, 0, 1)

        self.curr_agent_reputation[task_id][agent_idx] = new_reputation

        return new_reputation

    @staticmethod
    def calculate_agent_fitness(agent_reputation, agent_bid, alpha=0.5):
        """Derive agent fitness from a linear model, and move the score to logit space with exponential decay"""

        if not agent_reputation:
            return []

        agent_reputation = np.array(agent_reputation)
        agent_bid = np.array(agent_bid)

        V_q = agent_reputation**alpha
        utility = V_q - agent_bid  # Linear in price, as in original paper

        # Normalize scores so they sum to 1
        utility = utility / np.sum(utility)

        # for _agent_reputation, _agent_bid, _utility in zip(agent_reputation, agent_bid, utility):
        #     logger.debug(f"rep: {_agent_reputation:.4f}, price: {_agent_bid:.4f}, agent_score: {_utility:.4f}")

        return utility

    @staticmethod
    def utility_ces(rep_norm, price_norm, w_q=0.6, rho=0.0, eta=1.0):
        """q = g(rep), s = price_norm^{-eta}. CES aggregator with parameter rho.
        rho -> 0 yields Cobb–Douglas: U = q^{w_q} * s^{w_s}"""

        w_s = 1 - w_q

        rep_norm = rep_norm
        price_adj = price_norm ** (-eta)  # >1 discount, <1 premium
        if abs(rho) < 1e-8:
            U = (rep_norm**w_q) * (price_adj**w_s)  # Cobb–Douglas
        else:
            U = (w_q * (rep_norm**rho) + w_s * (price_adj**rho)) ** (1.0 / rho)
        S = U / (1.0 + U)  # (0,1)
        return U, S

    # topp douglas

    # # Cap at 110% of bidding price - bid lower is better
    # adj_price = max(1.1 - agent_bid, 0)

    # price_score = 1 - np.exp(-s_p * (adj_price))
    # reputation_score = 1 - np.exp(-s_r * (agent_reputation))

    # # Calculate a composite score here...
    # agent_score = reputation_score * a + price_score * (1 - a)

    # logger.debug(f"rep: {agent_reputation:.4f}, rep_score: {reputation_score:.4f}, price: {agent_bid:.4f}, price_score: {price_score:.4f}, agent_score: {agent_score:.4f}")

    # return agent_score

    def generate_listings(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Generate job payments from job definitions"""
        listings_by_task = {task_id: {} for task_id in self.task_ids}

        listings_by_job = {job_id: job.base_reward for job_id, job in self.jobs.items()}

        for job_id in self.job_ids:
            task_id = self.job_to_task_id[job_id]
            listings_by_task[task_id][job_id] = listings_by_job[job_id]

        return listings_by_task, listings_by_job

    def gumbel_rerank(self, fitness: np.ndarray, t=1) -> np.ndarray:
        """Efficient randomised ranking using Gumbel-Max trick."""
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(fitness))))

        # Original log probabilities (logits)
        log_probs = np.log(fitness)

        # Scale the logits by temperature BEFORE adding the noise
        reranked_agent_score = (log_probs / t) + gumbel_noise

        # Return indices sorted by perturbed skills (descending)
        return reranked_agent_score, np.argsort(reranked_agent_score)[::-1]

    def generate_market_preference(self, agent_pricing: List[Dict[str, float]]) -> Tuple[Dict[str, List[int]]]:
        """Create preference rankings for all JOBS based on agent pricing"""

        job_prefs: Dict[str, List[int]] = {}  # job_id -> ordered agent indices
        unranked_agent_scores: Dict[str, Dict[int, float]] = {}
        reranked_agent_scores: Dict[str, Dict[int, float]] = {}

        for job_id, job in self.jobs.items():
            task_id = job.task_id  # Get the skill type for this job

            agents_bidding = []
            bidding_agent_price = []
            bidding_agent_reputation = []

            for agent_idx, agent_job_price in enumerate(agent_pricing):
                if agent_bid := agent_job_price.get(job_id):  # Agents bid on jobs
                    # Use task-based reputation (skill reputation)
                    agent_reputation = self.curr_agent_reputation[task_id][agent_idx]
                    bidding_agent_reputation.append(agent_reputation)
                    bidding_agent_price.append(agent_bid)
                    agents_bidding.append(agent_idx)

            # Rest of the logic remains similar...
            unranked_agent_score = self.calculate_agent_fitness(bidding_agent_reputation, bidding_agent_price)

            if len(unranked_agent_score) >= 0:
                reranked_agent_score, job_ranking = self.gumbel_rerank(np.array(unranked_agent_score), t=self.t)
                job_prefs[job_id] = np.array(agents_bidding)[job_ranking]

                _reranked_agent_scores = {
                    agent_idx: float(reranked_task_agent_score)
                    for agent_idx, reranked_task_agent_score in zip(agents_bidding, reranked_agent_score)
                }
                _unranked_agent_scores = {
                    agent_idx: float(unranked_task_agent_score)
                    for agent_idx, unranked_task_agent_score in zip(agents_bidding, unranked_agent_score)
                }
                reranked_agent_scores[task_id] = _reranked_agent_scores
                unranked_agent_scores[task_id] = _unranked_agent_scores
            else:
                job_prefs[task_id] = []

        return job_prefs, unranked_agent_scores, reranked_agent_scores

    def match_jobs_multi(self, market_preference: Dict[str, List[int]]) -> Tuple[Dict[str, int], Set[int], List[str]]:
        job_matches = {}
        all_agents = set(np.arange(self.n_agents))
        unmatched_jobs = [job_id for job_id, market_pref in market_preference.items() if len(market_pref) == 0]
        matched_agents = set()

        for job_id, market_pref in market_preference.items():
            if len(market_pref) > 0:
                matched_agent_idx = market_pref[0]
                job_matches[job_id] = matched_agent_idx
                matched_agents.add(matched_agent_idx)

        return job_matches, all_agents - matched_agents, unmatched_jobs

    def match_jobs_single(
        self,
        agent_preferences: List[List[str]],
        market_preference: Dict[str, List[int]],
    ) -> Tuple[Dict[str, int], Set[int], List[str]]:
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
        job_current_match = {}  # {job_id: agent_id}
        job_agent_rank = {}  # {job_id: {agent_id: rank}}

        # Precompute agent rankings for each task - Maybe this should be a class. Will see
        for (
            job_id,
            agent_ranking,
        ) in market_preference.items():
            job_agent_rank[job_id] = {int(agent_idx): rank for rank, agent_idx in enumerate(agent_ranking)}

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
            job_id = agent_preferences[new_agent_idx][agent_next_proposal[new_agent_idx]]
            agent_next_proposal[new_agent_idx] += 1

            # If task is unmatched, accept proposal
            if job_id not in job_current_match:
                job_current_match[job_id] = new_agent_idx
            else:
                # Task is already matched, compare preferences
                matched_agent_idx = job_current_match[job_id]

                new_agent_rank = job_agent_rank[job_id][new_agent_idx]
                matched_agent_rank = job_agent_rank[job_id][matched_agent_idx]

                # Task prefers new agent if new agent has lower rank (higher preference)
                if (new_agent_rank >= 0) and (new_agent_rank < matched_agent_rank):
                    # Task switches to new agent
                    job_current_match[job_id] = new_agent_idx
                    free_agents.append(matched_agent_idx)  # Previous agent becomes free
                else:
                    # Task keeps current agent, new agent stays free
                    free_agents.append(new_agent_idx)

        unmatched_agents = set(range(n_agents)) - set(job_current_match.values())
        unmatched_jobs = [job_id for job_id, market_pref in market_preference.items() if len(market_pref) == 0]

        # Return matched, unmatched agents from agent perspective
        return job_current_match, unmatched_agents, unmatched_jobs

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

        self.agents[-1].get_agent_action(market_info)

        async def get_single_preference_with_retry(agent: AgentBase):
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, agent.get_agent_action, market_info),
                        timeout=retry_timeout if attempt > 0 else timeout_seconds,
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
                            f"Agent {agent.id} get_agent_action timed out after " f"{max_retries} retries. Giving up."
                        )
                        return AgentActionResponse(
                            action="error",
                            targets=[],
                            reasoning=f"TimeoutError: Agent action timed out after {max_retries} retries",
                        )

                except Exception as e:
                    if retry_on_exception and attempt < max_retries:
                        logger.warning(
                            f"Agent {agent.id} get_agent_action failed: "
                            f"{e.__class__.__name__}: {e}. Retrying... (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        logger.warning(f"Agent {agent.id} get_agent_action failed: " f"{e.__class__.__name__}: {e}")
                        return AgentActionResponse(action="error", targets=[], reasoning=f"{e.__class__.__name__}: {e}")

            # This should never be reached, but just in case
            return AgentActionResponse(action="error", targets=[], reasoning="Unknown error: max retries exceeded")

        # Create tasks for all agents
        async_tasks = [get_single_preference_with_retry(agent) for agent in self.agents]

        # Run all tasks concurrently
        try:
            agent_actions = await asyncio.gather(*async_tasks)
            return agent_actions
        except Exception as e:
            logger.warning(f"Batch agent call failed: {e.__class__.__name__}: {e}")
            return [
                AgentActionResponse(action="error", targets=[], reasoning=f"{e.__class__.__name__}: {e}")
                for _ in self.agents
            ]

    async def execute_jobs_async(
        self,
        job_matches: Dict[str, int],
        upgrade_skill_p=None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_timeout_seconds: Optional[float] = None,  # If None, uses same as timeout_seconds
        retry_on_exception: bool = False,  # Whether to retry on non-timeout exceptions
    ) -> Dict[str, Tuple[int, float]]:
        """Returns a tuple of job_id, agent_idx, agent_performance"""

        upgrade_skill_p = upgrade_skill_p or self.p
        retry_timeout = retry_timeout_seconds or timeout_seconds

        async def get_single_agent_performance_with_retry(runner: TaskRunner, task_id: str, agent_idx: int):
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, runner.perform_task, upgrade_skill_p),
                        timeout=retry_timeout if attempt > 0 else timeout_seconds,
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
                            f"Task (agent_idx:{agent_idx}, task_id:{task_id}) failed: " f"{e.__class__.__name__}: {e}"
                        )
                        return 0

            # This should never be reached, but just in case
            return 0

        if not job_matches:
            return {}

        job_ids, agent_ids = zip(*job_matches.items())
        jobs = [
            get_single_agent_performance_with_retry(
                self.task_runners[self.job_to_task_id[job_id]][agent_idx], job_id, agent_idx
            )
            for job_id, agent_idx in zip(job_ids, agent_ids)
        ]

        # try:
        job_performances = await asyncio.gather(*jobs, return_exceptions=True)
        # Recollapse to dict based on task_ids for easier retrieval
        return {job_id: (agent_idx, perf) for job_id, agent_idx, perf in zip(job_ids, agent_ids, job_performances)}

        # except Exception as e:
        #     logger.warning(f"Batch task runner call failed: {e.__class__.__name__}: {e}")
        #     return {}

    def _collect_agent_bids(self, market_info: MarketInfo) -> Tuple[List[AgentActionResponse], Dict]:
        """Collect and process agent bids from agent actions"""
        # Get agent responses
        agent_responses: List[AgentActionResponse] = asyncio.run(self.get_agent_actions_async(market_info))

        # Process bids
        agent_bidding_data = {"preferences": [], "pricing": [], "pricing_normalized": []}

        for agent_response in agent_responses:
            if agent_response.action == "bid":
                job_bids = {job_id: price for job_id, price in agent_response.targets}

                agent_bidding_data["pricing"].append(job_bids)
                agent_bidding_data["pricing_normalized"].append(
                    {
                        job_id: price / market_info.listings[self.job_to_task_id[job_id]][job_id]
                        for job_id, price in job_bids.items()
                    }
                )
                agent_bidding_data["preferences"].append([job_id for job_id, _ in agent_response.targets])
            else:
                agent_bidding_data["pricing"].append({})
                agent_bidding_data["pricing_normalized"].append({})
                agent_bidding_data["preferences"].append([])

        return agent_responses, agent_bidding_data

    def _train_unmatched_agents(
        self, unmatched_agents: Set[int], agent_responses: List[AgentActionResponse]
    ) -> Dict[int, str]:
        """Handle skill training for unmatched agents"""
        training_performed = {}
        for agent_idx in unmatched_agents:

            agent_action: AgentActionResponse = agent_responses[agent_idx]

            if len(agent_action.targets) == 0:

                logger.warning(f"Empty agent action for agent {agent_idx}: {agent_action}")

                return ""

            if agent_action.action == "train":
                task_id = agent_action.targets[0][0]

            elif agent_action.action == "bid":
                task_id = self.job_to_task_id[agent_action.targets[0][0]]

            agent_task_runner = self.task_runners[task_id][agent_idx]

            # For unmatched agents, upgrade their skills here
            # TODO: Separate train and bid? Make lost agents "win" easier??
            if agent_action.action == "train" or (agent_action.action == "bid" and (np.random.uniform(0, 1) <= self.p)):

                print(self.curr_agent_reputation[task_id][agent_idx])
                agent_task_runner.upgrade_skill()
                training_performed[agent_idx] = task_id

                # Benchmark agent without updating to history
                benchmark_performance = agent_task_runner.perform_task(upgrade_skill_p=0)
                reputation = self.update_reputation(agent_idx, task_id, benchmark_performance)

                print(reputation)

        return training_performed

    def _process_job_performances(
        self,
        listings_by_job: Dict[str, float],
        agent_bidding_data: Dict,
        job_matches: Dict[str, int],
        job_performances: Dict[str, Tuple[int, float]],
    ) -> Dict:
        """Process results and update reputation"""

        # Initialize tracking structures
        agent_round_rewards = [0.0] * self.n_agents
        agent_allocations = [[] for _ in range(self.n_agents)]  # List of allocations per agent
        agent_reputation_updates = [{} for _ in range(self.n_agents)]
        winning_bid_prices = {}

        # Process each matched job
        for job_id, agent_idx in job_matches.items():
            if job_id not in job_performances:
                continue

            _agent_idx, performance = job_performances[job_id]
            assert agent_idx == _agent_idx, f"Agent mismatch for job {job_id}"

            # Get task type for this job
            task_id = self.job_to_task_id[job_id]

            # Update reputation for the skill
            old_reputation = self.curr_agent_reputation[task_id][agent_idx]
            new_reputation = self.update_reputation(agent_idx, task_id, performance)

            agent_reputation_updates[agent_idx][task_id] = new_reputation

            # Calculate rewards
            base_price = listings_by_job[job_id]
            bid_price = agent_bidding_data["pricing"][agent_idx][job_id]
            adjusted_reward = bid_price * performance

            # Track job allocation for individual agents
            job_history = JobHistory(
                job_id=job_id,
                task_id=task_id,
                base_price=base_price,
                bid_price=bid_price,
                performance=performance,
                adjusted_reward=adjusted_reward,
                old_reputation=old_reputation,
                new_reputation=new_reputation,
            )
            agent_allocations[agent_idx].append(job_history)

            # Update totals
            agent_round_rewards[agent_idx] += adjusted_reward
            winning_bid_prices[job_id] = bid_price

            # Store performance
            perf_record = AgentPerformance(
                agent_idx=agent_idx,
                agent_id=self.agent_ids[agent_idx],
                round=self.round_counter,
                job_id=job_id,
                task_id=task_id,
                performance=performance,
            )

            self.job_performance.append(perf_record)

        return {
            "agent_round_rewards": agent_round_rewards,
            "agent_allocations": agent_allocations,
            "agent_reputation_updates": agent_reputation_updates,
            "winning_bid_prices": winning_bid_prices,
            "job_performances": job_performances,
        }

    def _send_agent_feedback(
        self,
        round_results: Dict,
        training_performed: Dict,
        agent_responses: List[AgentActionResponse],
        listings: Dict[str, float],
    ):
        """Send feedback to all agents about their round results"""
        for agent_idx, agent in enumerate(self.agents):
            # Build agent history with multiple allocations
            agent_action = agent_responses[agent_idx]
            target_jobs = [t[0] for t in agent_action.targets] if agent_action.action == "bid" else []
            allocated_jobs: List[JobHistory] = round_results["agent_allocations"][agent_idx]
            unallocated_jobs = list(set(target_jobs) - set(a.job_id for a in allocated_jobs))
            agent_history = AgentHistory(
                round=self.round_counter,
                agent_action=agent_responses[agent_idx],
                listings=listings,
                allocated_jobs=allocated_jobs,
                unallocated_jobs=unallocated_jobs,
                total_reward=round_results["agent_round_rewards"][agent_idx],
                reputation_update={task_id: rep[agent_idx] for task_id, rep in self.curr_agent_reputation.items()},
                training_performed=training_performed.get(agent_idx, ""),
            )

            agent.receive_response(agent_history)

    def _format_bids_by_job(self, agent_pricing: List[Dict]) -> Dict[str, Dict[int, float]]:
        """Format agent bids indexed by job_id"""
        return {
            job_id: {agent_idx: pricing[job_id] for agent_idx, pricing in enumerate(agent_pricing) if job_id in pricing}
            for job_id in self.job_ids
        }

    def simulate_timestep(self) -> None:
        """Simulate one timestep of the market"""
        self.round_counter += 1

        # 1. Prepare market information
        market_history_string = self.get_market_history_string()

        listings_by_task, listings_by_job = self.generate_listings()

        market_info = MarketInfo(
            round=self.round_counter,
            history=market_history_string,
            listings=listings_by_task,
            info={
                "round_data": self.history[-10:] if self.history else None,
                "agent_skills": {agent.id: agent.skill_level_by_task for agent in self.agents},
            },
        )

        # Store previous reputation
        prev_reputation = deepcopy(self.curr_agent_reputation)

        # 2. Collect agent bids
        agent_responses, agent_bidding_data = self._collect_agent_bids(market_info)

        # 3. Generate market preferences and match
        market_preference, unranked_agent_scores, reranked_agent_scores = self.generate_market_preference(
            agent_bidding_data["pricing_normalized"]
        )

        # job_matches, unmatched_agents, unmatched_jobs = self.match_jobs_single(agent_bidding_data["preferences"], market_preference)
        job_matches, unmatched_agents, unmatched_jobs = self.match_jobs_multi(market_preference)

        # 4. Execute matched jobs and collect performance
        job_performances = asyncio.run(self.execute_jobs_async(job_matches))

        # 5. Process results and update state
        round_results = self._process_job_performances(
            listings_by_job=listings_by_job,
            agent_bidding_data=agent_bidding_data,
            job_matches=job_matches,
            job_performances=job_performances,
        )

        # 6. Handle unmatched agents training
        training_performed = self._train_unmatched_agents(unmatched_agents, agent_responses)

        # 7. Send feedback to agents
        self._send_agent_feedback(round_results, training_performed, agent_responses, listings_by_job)

        # 8. Store round history
        # Calculate cumulative rewards
        if self.history:
            agent_total_rewards = list(
                np.array(self.history[-1].agent_total_rewards) + np.array(round_results["agent_round_rewards"])
            )
        else:
            agent_total_rewards = round_results["agent_round_rewards"]

        # Create round data
        round_data = RoundData(
            round=self.round_counter,
            base_prices=listings_by_job,
            agent_actions=agent_responses,
            agent_bids=self._format_bids_by_job(agent_bidding_data["pricing"]),
            agent_bids_normalized=self._format_bids_by_job(agent_bidding_data["pricing_normalized"]),
            agent_preferences=agent_bidding_data["preferences"],
            winning_prices=round_results["winning_bid_prices"],
            unranked_agent_scores=unranked_agent_scores,
            reranked_agent_scores=reranked_agent_scores,
            market_preference=market_preference,
            matched_jobs=job_matches,
            unmatched_agents=unmatched_agents,
            unmatched_jobs=unmatched_jobs,
            prev_reputation=prev_reputation,
            agent_reputation=deepcopy(self.curr_agent_reputation),
            agent_skills=[agent.skill_level_by_task for agent in self.agents],
            job_performance=job_performances,
            agent_round_rewards=round_results["agent_round_rewards"],
            agent_total_rewards=agent_total_rewards,
        )

        self.history.append(round_data)

    # def simulate_timestep(self) -> Dict:
    #     """
    #     Simulate one timestep of the market
    #     """
    #     self.round_counter += 1

    #     # market_history_string = self.get_market_history_string()

    #     # # Generate task reward as listings
    #     # base_job_rewards = self.generate_job_payments()

    #     # market_info = MarketInfo(
    #     #     round=self.round_counter,
    #     #     history=market_history_string,
    #     #     listings=base_job_rewards,
    #     #     info={
    #     #         "round_data": self.history[-10:] if self.history else None,
    #     #         "agent_skills": {agent.id: agent.skill_level_by_task for agent in self.agents},
    #     #     },
    #     # )

    #     # # Get agent data via API
    #     # agent_responses: List[AgentActionResponse] = asyncio.run(self.get_agent_actions_async(market_info))

    #     # agent_job_preference: List[List[str]] = []
    #     # agent_job_pricing: List[Dict[str, float]] = []
    #     # agent_job_pricing_normalized: List[Dict[str, float]] = []

    #     # for agent_response in agent_responses:

    #     #     if agent_response.action == "bid":

    #     #         agent_job_pricing.append({job_id: agent_price for job_id, agent_price in agent_response.targets})
    #     #         agent_job_pricing_normalized.append(
    #     #             {job_id: agent_price / base_job_rewards[job_id] for job_id, agent_price in agent_response.targets}
    #     #         )
    #     #         agent_job_preference.append([job_id for job_id, _ in agent_response.targets])

    #     #     else:
    #     #         agent_job_pricing.append({})
    #     #         agent_job_pricing_normalized.append({})
    #     #         agent_job_preference.append([])

    #     # Create market preferences per task based on agent skill level and bids

    #     # Run stable matching algorithm (or single)

    #     # Base reward of task for winning bid before performance adjustment
    #     winning_bid_prices: Dict[str, float] = {}

    #     # Round reward by agent
    #     agent_round_rewards = [0] * self.n_agents

    #     job_performance_dict = {}

    #     prev_reputation = deepcopy(self.curr_agent_reputation)

    #     for job_id, agent_idx in job_matches.items():

    #         # Check if performing agent is not the same (shouldn't ever happen)
    #         _agent_idx, performance = job_performance_dict[job_id]
    #         assert agent_idx == _agent_idx

    #         # Update agent reputation per agent / task
    #         task_id = self.job_to_task_id[job_id]
    #         new_reputation = self.update_reputation(agent_idx, task_id, performance)

    #         # Get agent bid price, and update agent's adjusted reward based on its bid price * its performance
    #         agent_bid_price = agent_job_pricing[agent_idx][job_id]
    #         adjusted_reward = agent_bid_price * performance

    #         # Log reward
    #         winning_bid_prices[job_id] = agent_bid_price
    #         agent_round_rewards[agent_idx] += adjusted_reward

    #         agent_performance = AgentPerformance(
    #             agent_idx=agent_idx,
    #             agent_id=self.agent_ids[agent_idx],
    #             round=self.round_counter,
    #             task_id=task_id,
    #             job_id=job_id,
    #             performance=performance,
    #         )

    #         self.job_performance.append(agent_performance)

    #         market_response = AgentHistory(
    #             round=self.round_counter,
    #             allocated=job_id,
    #             agent_action=agent_responses[agent_idx],
    #             listings=base_job_rewards,
    #             agent_bid_price=agent_bid_price,
    #             agent_performance=performance,
    #             adjusted_reward=adjusted_reward,
    #             reputation=new_reputation,
    #         )

    #         self.agents[agent_idx].receive_response(market_response)

    #     for agent_idx in unmatched_agents:

    #         agent_action: AgentActionResponse = agent_responses[agent_idx]

    #         if agent_action.targets:

    #             # Get the first task_id in agent's order of preference
    #             first_task_id = agent_action.targets[0][0]
    #             agent_action = agent_action.action

    #         else:

    #             logger.warning(f"Empty agent action for agent {agent_idx}: {agent_action}")

    #             first_task_id = "None"

    #         agent_task_runner = self.task_runners[first_task_id][agent_idx]

    #         # For unmatched agents, upgrade their skills here
    #         if agent_action == "train":
    #             agent_task_runner.upgrade_skill()
    #         elif agent_action == "bid" and (np.random.uniform(0, 1) <= self.p):
    #             agent_task_runner.upgrade_skill()

    #         market_response = AgentHistory(
    #             round=self.round_counter,
    #             allocated=first_task_id,
    #             listings=base_job_rewards,
    #             agent_action=agent_responses[agent_idx],
    #         )

    #         self.agents[agent_idx].receive_response(market_response)

    #     if self.history:
    #         agent_total_rewards = list(np.array(self.history[-1].agent_total_rewards) + np.array(agent_round_rewards))
    #     else:
    #         agent_total_rewards = agent_round_rewards

    #     # Just for logging

    #     agent_bids: Dict[str, Dict[int, float]] = {
    #         task_id: {
    #             agent_idx: agent_job_bid[task_id]
    #             for agent_idx, agent_job_bid in enumerate(agent_job_pricing)
    #             if agent_job_bid.get(task_id) is not None
    #         }
    #         for task_id in self.task_ids
    #     }
    #     agent_bids_normalized: Dict[str, Dict[int, float]] = {
    #         task_id: {
    #             agent_idx: agent_job_bid[task_id]
    #             for agent_idx, agent_job_bid in enumerate(agent_job_pricing_normalized)
    #             if agent_job_bid.get(task_id) is not None
    #         }
    #         for task_id in self.task_ids
    #     }

    #     round_data = RoundData(
    #         round=self.round_counter,
    #         agent_actions=agent_responses,
    #         agent_bids=agent_bids,
    #         agent_bids_normalized=agent_bids_normalized,
    #         agent_preferences=agent_job_preference,
    #         base_rewards=base_job_rewards,
    #         task_rewards=winning_bid_prices,
    #         prev_reputation=prev_reputation,
    #         agent_reputation=deepcopy(self.curr_agent_reputation),
    #         agent_skills=[agent.skill_level_by_task for agent in self.agents],
    #         unranked_agent_scores=unranked_agent_scores,
    #         reranked_agent_scores=reranked_agent_scores,
    #         market_preference=market_preferences,
    #         matched_task_agent=job_matches,
    #         unmatched_agents=unmatched_agents,
    #         task_performance=job_performance_dict,
    #         agent_round_rewards=agent_round_rewards,
    #         agent_total_rewards=agent_total_rewards,
    #     )

    #     self.history.append(round_data)

    def get_market_history_string(self, n_steps=10) -> str:
        """Generate formatted history string matching your example format"""

        if not self.history:
            return "This is Round 1. No history recorded yet."

        lines = []

        for round_data in self.history[-n_steps:]:
            allocations = []
            for job_id in sorted(round_data.matched_jobs):
                task_id = self.job_to_task_id[job_id]
                agent_idx = round_data.matched_jobs[job_id]
                agent_name = self.agents[agent_idx].id
                rep = round(round_data.prev_reputation[task_id][agent_idx], 2)
                price = round_data.base_prices[job_id]
                allocations.append(f"{job_id}@{price}→{agent_name}({rep})")

            lines.append(f"R{round_data.round}: {' | '.join(allocations)}")

        agent_rewards = self.history[-1].agent_total_rewards

        reward_sorted = " ".join(
            [
                f"#{i + 1}: {self.agent_ids[agent_idx]}, ${agent_rewards[agent_idx]:.1f}"
                for i, agent_idx in enumerate(np.argsort(agent_rewards)[::-1])
            ]
        )

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

    def export(self, filepath=None) -> ExperimentLog:

        config = dict(
            p=self.p,
            t=self.t,
            rep_initial=self.rep_initial,
            rep_window=self.rep_window,
            rep_sensitivity=self.rep_sensitivity,
            rep_lambda=self.rep_lambda,
        )

        exp_log = ExperimentLog(
            config=config,
            agent_ids=self.agent_ids,
            job_ids=self.job_ids,
            task_ids=self.task_ids,
            job_to_task_id=self.job_to_task_id,
            history=self.history,
            job_performance=self.job_performance,
            agents=[agent.export() for agent in self.agents],
            token_usage=self.get_token_usage(),
        )

        if filepath:
            with open(filepath, "w") as f:
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
