# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any, Literal
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
from ssa.common import (
    Job,
    ExperimentLog,
    AgentLog,
    RoundData,
    JobHistory,
    AgentHistory,
    AgentPerformance,
    AgentActionResponse,
    MarketInfo,
)
from ssa.galeshapley import multi_galeshapley

from ssa.tasks.cipher import CipherTask

import nest_asyncio

# Add this at the top of your notebook/script
nest_asyncio.apply()

EWMA_SPAN = 5
REPUTATION_PRIOR_STRENGTH = 1
INITIAL_REPUTATION = 0.5
GUMBEL_NOISE = 0.2
SKILL_P = 0.2
LAMBDA = 0.85
MARKET_LIMIT = 1
HISTORY_LIMIT = 10
MARKET_PREF_LIMIT = 10
AGENT_PREF_LIMIT = 5


class LabourMarket:
    def __init__(
        self,
        jobs: List[Job],
        tasks: List[TaskBase],
        agents: List[AgentBase],
        market_limit=MARKET_LIMIT,
        market_pref_limit=MARKET_PREF_LIMIT,
        agent_pref_limit=AGENT_PREF_LIMIT,
        history_limit=HISTORY_LIMIT,
        skill_phi=SKILL_P,
        rep_initial=INITIAL_REPUTATION,
        rep_window=EWMA_SPAN,
        rep_sensitivity=REPUTATION_PRIOR_STRENGTH,
        rep_lambda=LAMBDA,
        gumbel_t=GUMBEL_NOISE,
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

        self.market_limit = market_limit
        self.market_pref_limit = market_pref_limit
        self.agent_pref_limit = agent_pref_limit
        self.history_limit = history_limit

        # Inject other agent info to agent class
        for agent_idx, agent in enumerate(self.agents):
            agent.idx = agent_idx
            agent.agent_ids = self.agent_ids

        self.gumbel_t = gumbel_t
        self.skill_phi = skill_phi

        # Initialize data tracking
        self.history: List[RoundData] = []
        self.round_counter = 0

        # Initialize task runners
        self.task_runners = {
            task.id: [TaskRunner(task=deepcopy(task), subagent=agent.subagents[task.id]) for agent in agents]
            for task in tasks
        }

        # Initialize task reputation by agent
        self.rep_initial = rep_initial
        self.rep_window = rep_window
        self.rep_sensitivity = rep_sensitivity
        self.rep_lambda = rep_lambda
        self.curr_agent_reputation = {task_id: [self.rep_initial] * self.n_agents for task_id in self.task_ids}

        logger.info(
            f"Set up LabourMarket with {self.n_agents} agents and {self.n_jobs} jobs over {self.n_tasks} tasks. Initializing..."
        )
        self.initialize()

    def get_job_performance(self, agent_idx=None, agent_id=None, task_id=None, job_id=None, filter_initial=False):
        return [
            perf
            for perf in self.job_performance
            if (agent_idx is None or perf.agent_idx == agent_idx)
            and (agent_id is None or perf.agent_id == agent_id)
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
                    reputation=-1,
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
    def utility_che(agent_reputation, agent_bid, alpha=0.5):
        """Derive agent fitness from a linear model, and move the score to logit space with exponential decay"""

        if not agent_reputation:
            return []

        agent_reputation = np.array(agent_reputation)
        agent_bid = np.array(agent_bid)

        V_q = agent_reputation**alpha
        utility = V_q - agent_bid  # Linear in price, as in original paper

        # Normalize scores so they sum to 1
        utility = utility / (np.sum(utility) + 1e-9)

        return utility

    @staticmethod
    def utility_ces(rep_norm, price_norm, w_q=0.6, rho=0.0, eta=1.0):
        """q = g(rep), s = price_norm^{-eta}. CES aggregator with parameter rho.
        rho -> 0 yields Cobb–Douglas: U = q^{w_q} * s^{w_s}"""

        rep_norm = np.array(rep_norm)
        price_norm = np.array(price_norm)

        w_s = 1 - w_q

        rep_norm = rep_norm
        price_adj = price_norm ** (-eta)  # >1 discount, <1 premium
        if abs(rho) < 1e-8:
            utility = (rep_norm**w_q) * (price_adj**w_s)  # Cobb–Douglas
        else:
            utility = (w_q * (rep_norm**rho) + w_s * (price_adj**rho)) ** (1.0 / rho)
        utility_score = utility / (1.0 + utility)  # (0,1)
        return utility_score

    @staticmethod
    def gumbel_rerank(score: np.ndarray, t=1) -> Tuple[np.ndarray, np.ndarray]:
        """Efficient randomised ranking using Gumbel-Max trick."""

        if t < 1e-9:
            return score, np.argsort(score)[::-1]

        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(score))))

        # Original log probabilities (logits)
        log_probs = np.log(score)

        # Scale the logits by temperature BEFORE adding the noise
        reranked_score = (log_probs / t) + gumbel_noise

        # Return indices sorted by perturbed skills (descending)
        return reranked_score, np.argsort(reranked_score)[::-1]

    def generate_listings(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Generate job payments from job definitions"""
        listings_by_task = {task_id: {} for task_id in self.task_ids}

        listings_by_job = {job_id: job.base_reward for job_id, job in self.jobs.items() if job.get_base_reward() >= 0}

        for job_id in listings_by_job.keys():
            task_id = self.job_to_task_id[job_id]
            listings_by_task[task_id][job_id] = listings_by_job[job_id]

        return listings_by_task, listings_by_job

    def generate_market_preference(
        self, listings_by_job, agent_pricing: List[Dict[str, float]]
    ) -> Tuple[Dict[str, List[int]]]:
        """Create preference rankings for all JOBS based on agent pricing"""

        job_prefs: Dict[str, List[int]] = {}  # job_id -> ordered agent indices
        unranked_agent_scores: Dict[str, Dict[int, float]] = {}
        reranked_agent_scores: Dict[str, Dict[int, float]] = {}

        for job_id in listings_by_job:
            job = self.jobs[job_id]
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
            unranked_agent_score = self.utility_ces(bidding_agent_reputation, bidding_agent_price)

            if len(unranked_agent_score) > 0:
                reranked_agent_score, job_ranking = self.gumbel_rerank(np.array(unranked_agent_score), t=self.gumbel_t)
                job_prefs[job_id] = np.array(agents_bidding)[job_ranking][: self.market_pref_limit]

                _reranked_agent_scores = {
                    agent_idx: float(reranked_task_agent_score)
                    for agent_idx, reranked_task_agent_score in zip(agents_bidding, reranked_agent_score)
                }
                _unranked_agent_scores = {
                    agent_idx: float(unranked_task_agent_score)
                    for agent_idx, unranked_task_agent_score in zip(agents_bidding, unranked_agent_score)
                }
                reranked_agent_scores[job_id] = _reranked_agent_scores
                unranked_agent_scores[job_id] = _unranked_agent_scores
            else:
                job_prefs[task_id] = []

        return job_prefs, unranked_agent_scores, reranked_agent_scores

    def _match_jobs_all(self, market_preference: Dict[str, List[int]]) -> Tuple[Dict[str, int], Set[int], List[str]]:
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

    async def _get_agent_actions_async(
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

        upgrade_skill_p = upgrade_skill_p or self.skill_phi
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

    def filter_agent_response(self, response: AgentActionResponse) -> AgentActionResponse:
        """Filters response.targets to only include valid job_ids or skill_ids"""
    
        if response.action == "bid":
            valid_targets = [target for target in response.targets if target[0] in self.job_ids]
        elif response.action == "train":
            valid_targets = [target for target in response.targets if target[0] in self.task_ids]
        else:
            valid_targets = response.targets  # No filtering for "error"
        
        return AgentActionResponse(
            reasoning=response.reasoning,
            action=response.action,
            targets=valid_targets
    )


    def collect_agent_bids(self, market_info: MarketInfo) -> Tuple[List[AgentActionResponse], Dict]:
        """Collect and process agent bids from agent actions"""
        # Get agent responses
        agent_responses: List[AgentActionResponse] = asyncio.run(self._get_agent_actions_async(market_info))
        agent_responses = [self.filter_agent_response(response) for response in agent_responses]

        listings_by_job = {}
        for listings_by_task in market_info.listings.values():
            for job_id, base_price in listings_by_task.items():
                listings_by_job[job_id] = base_price

        # Process bids
        agent_bidding_data = {"preferences": [], "pricing": [], "pricing_normalized": []}

        for agent_response in agent_responses:
            if agent_response.action == "bid":
                job_bids = {}
                for job_id, price in agent_response.targets:
                    if job_id in listings_by_job:
                        try:
                            job_bids[job_id] = float(price)
                        except (ValueError, TypeError):
                            job_bids[job_id] = 0

                agent_bidding_data["pricing"].append(job_bids)
                agent_bidding_data["pricing_normalized"].append(
                    {
                        job_id: price / listings_by_job.get(job_id, price) if price != 0 else 0
                        for job_id, price in job_bids.items()
                    }
                )
                agent_bidding_data["preferences"].append(
                    [job_id for job_id, _ in agent_response.targets][: self.agent_pref_limit]
                )

            else:
                agent_bidding_data["pricing"].append({})
                agent_bidding_data["pricing_normalized"].append({})
                agent_bidding_data["preferences"].append([])

        return agent_responses, agent_bidding_data

    def train_agents(self, agents: Set[int], agent_responses: List[AgentActionResponse]) -> Dict[int, str]:
        """Handle skill training for unmatched agents"""
        training_performed = {}
        for agent_idx in agents:

            agent_action: AgentActionResponse = agent_responses[agent_idx]

            if len(agent_action.targets) == 0:

                logger.warning(f"Empty agent action for agent {agent_idx}: {agent_action}")

                return {}

            if agent_action.action == "train":
                task_id = agent_action.targets[0][0]

            elif agent_action.action == "bid":
                task_id = self.job_to_task_id[agent_action.targets[0][0]]

            agent_task_runner = self.task_runners[task_id][agent_idx]

            # For unmatched agents, upgrade their skills here
            # TODO: Separate train and bid? Make lost agents "win" easier??
            if agent_action.action == "train" or (
                agent_action.action == "bid" and (np.random.uniform(0, 1) <= self.skill_phi)
            ):

                agent_task_runner.upgrade_skill()
                training_performed[agent_idx] = task_id

            # Benchmark agent without updating to history
            benchmark_performance = agent_task_runner.perform_task(upgrade_skill_p=0, benchmark=True)
            reputation = self.update_reputation(agent_idx, task_id, benchmark_performance)

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
                reputation=new_reputation,
            )

            self.job_performance.append(perf_record)

        return {
            "agent_round_rewards": agent_round_rewards,
            "agent_allocations": agent_allocations,
            "agent_reputation_updates": agent_reputation_updates,
            "winning_bid_prices": winning_bid_prices,
            "job_performances": job_performances,
            "listings": listings_by_job,
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
            target_jobs = {k: v for k, v in agent_action.targets} if agent_action.action == "bid" else {}
            allocated_jobs: List[JobHistory] = round_results["agent_allocations"][agent_idx]
            unallocated_job_ids = set(target_jobs.keys()) - set(a.job_id for a in allocated_jobs)
            unallocated_jobs = [
                JobHistory(
                    job_id=job_id,
                    task_id=self.job_to_task_id.get(job_id, f"ERROR ({job_id})"),
                    base_price=round_results["listings"].get(job_id, 0),
                    bid_price=target_jobs[job_id],
                )
                for job_id in unallocated_job_ids
            ]
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

        listings_by_task, listings_by_job = self.generate_listings()

        market_info = MarketInfo(
            round=self.round_counter,
            round_info=self.history[-self.history_limit :] if self.history else [],
            listings=listings_by_task,
            info={
                "agent_skills": {agent.id: agent.skill_level_by_task for agent in self.agents},
            },
        )

        # Store previous reputation
        prev_reputation = deepcopy(self.curr_agent_reputation)

        # 2. Collect agent bids
        agent_responses, agent_bidding_data = self.collect_agent_bids(market_info)

        # 3. Generate market preferences and match
        market_preference, unranked_agent_scores, reranked_agent_scores = self.generate_market_preference(
            listings_by_job,
            agent_bidding_data["pricing_normalized"],
        )

        job_matches, unmatched_agents, unmatched_jobs = multi_galeshapley(
            agent_bidding_data["preferences"], market_preference, multi_limit=self.market_limit
        )

        # job_matches, unmatched_agents, unmatched_jobs = self._match_jobs_all(market_preference)

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
        training_performed = self.train_agents(unmatched_agents, agent_responses)

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

        return self.generate_market_summary(round_data)

    def generate_market_summary(self, round_data: RoundData):
        """Generate formatted history string matching your example format"""

        allocations = []
        for job_id in sorted(round_data.matched_jobs):
            task_id = self.job_to_task_id[job_id]
            agent_idx = round_data.matched_jobs[job_id]
            agent_name = self.agent_ids[agent_idx]
            rep = round(round_data.prev_reputation[task_id][agent_idx] * 5, 1)
            price = round_data.base_prices[job_id]
            allocations.append(f"{job_id}(${price})→{agent_name}({rep}*)")

        agent_rewards = round_data.agent_total_rewards
        reward_sorted = " ".join(
            [
                f"#{i + 1}: {self.agent_ids[agent_idx]}, ${agent_rewards[agent_idx]:.1f}"
                for i, agent_idx in enumerate(np.argsort(agent_rewards)[::-1])
            ]
        )

        summary_str = f"R{round_data.round}: {reward_sorted}\n{', '.join(allocations)}"
        return summary_str

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
            market_limit=self.market_limit,
            market_pref_limit=self.market_pref_limit,
            agent_pref_limit=self.agent_pref_limit,
            history_limit=self.history_limit,
            skill_phi=self.skill_phi,
            rep_initial=self.rep_initial,
            rep_window=self.rep_window,
            rep_sensitivity=self.rep_sensitivity,
            rep_lambda=self.rep_lambda,
            gumbel_t=self.gumbel_t,
        )

        jobs = [j.model_dump() for j in self.jobs.values()]

        exp_log = ExperimentLog(
            config=config,
            jobs=jobs,
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


# %%
