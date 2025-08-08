# %%
import numpy as np
from typing import List, Dict, Optional


class LaborMarket:
    def __init__(self, n_tasks: int):
        self.n_tasks = n_tasks

    def generate_tasks(self) -> np.ndarray:
        """Generate random task payments"""
        return [10, 10, 10]
        return np.random.uniform(1, 10, self.n_tasks)

    def skill_weighted_ranking(self, skills: np.ndarray) -> np.ndarray:
        """Efficient skill-weighted ranking without repeated sampling"""

        # return np.argsort(-skills)
        uniform_random = np.random.uniform(0, 1, len(skills))

        scores = uniform_random ** (1.0 / skills)

        return np.argsort(-scores)  # Descending order (best first)

    def create_task_preferences(self, agent_skills: np.ndarray) -> List[np.ndarray]:
        """
        Create preference rankings for all tasks based on agent skills

        Args:
            agent_skills: Shape (n_agents, n_tasks) - each agent's skill for each task

        Returns:
            List of length n_tasks, each containing agent ranking for that task
        """
        task_prefs = []
        for task_idx in range(self.n_tasks):
            relevant_skills = agent_skills[:, task_idx]

            task_ranking = self.skill_weighted_ranking(relevant_skills)
            task_prefs.append(task_ranking)
        return task_prefs

    def gale_shapley(
        self, agent_preferences: List[np.ndarray], task_preferences: List[np.ndarray]
    ) -> Dict[int, int]:
        """
        Gale-Shapley matching with agents proposing to tasks

        Args:
            agent_preferences: List of length n_agents, each containing task ranking
            task_preferences: List of length n_tasks, each containing agent ranking

        Returns:
            Dict {agent_id: task_id} of matches
        """
        n_agents = len(agent_preferences)

        # Initialize data structures
        agent_next_proposal = np.zeros(
            n_agents, dtype=int
        )  # Next task index to propose to
        task_current_match = {}  # {task_id: agent_id}
        task_agent_rank = {}  # {task_id: {agent_id: rank}}

        # Precompute agent rankings for each task for O(1) lookup
        for task_id, agent_ranking in enumerate(task_preferences):
            task_agent_rank[task_id] = {
                agent_id: rank for rank, agent_id in enumerate(agent_ranking)
            }

        # Track free agents
        free_agents = set(range(n_agents))

        while free_agents:
            # Pick any free agent
            agent = free_agents.pop()

            # Check if agent has exhausted all tasks
            if agent_next_proposal[agent] >= self.n_tasks:
                continue  # Agent remains unmatched

            # Agent proposes to next preferred task
            task = agent_preferences[agent][agent_next_proposal[agent]]
            agent_next_proposal[agent] += 1

            # If task is unmatched, accept proposal
            if task not in task_current_match:
                task_current_match[task] = agent
            else:
                # Task is already matched, compare preferences
                current_agent = task_current_match[task]

                # Task prefers new agent if new agent has lower rank (higher preference)
                if task_agent_rank[task][agent] < task_agent_rank[task][current_agent]:
                    # Task switches to new agent
                    task_current_match[task] = agent
                    free_agents.add(current_agent)  # Previous agent becomes free
                else:
                    # Task keeps current agent, new agent stays free
                    free_agents.add(agent)

        # Return matching from agent perspective
        return {agent: task for task, agent in task_current_match.items()}

    def simulate_timestep(self, agents) -> Dict:
        """
        Simulate one timestep of the market

        Args:
            agents: List of agent objects with get_preferences() and get_skills() methods

        Returns:
            Dict with matching results and metadata
        """
        # Generate tasks and payments
        payments = self.generate_tasks()

        # Get agent data via API
        agent_preferences = [agent.get_preferences() for agent in agents]
        agent_skills = np.array(
            [agent.get_skills() for agent in agents]
        )  # Shape: (n_agents, n_tasks)

        # Create task preferences based on skills
        task_preferences = self.create_task_preferences(agent_skills)

        # Run matching
        matches = self.gale_shapley(agent_preferences, task_preferences)

        # Calculate rank-adjusted payments
        adjusted_payments = {}
        agent_ranks = {}
        task_ranks = {}  # Separate variable for task ranks

        for agent_id, task_id in matches.items():
            # Find this agent's rank in the task's preference list (how much task wants agent)
            task_ranking = task_preferences[
                task_id
            ]  # Array of agent IDs in preference order
            agent_rank = np.where(task_ranking == agent_id)[0][0]  # 0-indexed position

            # Find this task's rank in the agent's preference list (how much agent wants task)
            agent_ranking = agent_preferences[
                agent_id
            ]  # Array of task IDs in preference order
            task_rank = np.where(agent_ranking == task_id)[0][0]  # 0-indexed position

            # Get agent's skill for this specific task
            agent_skill = agent_skills[agent_id, task_id]

            # Calculate adjusted payment based on ranks AND skill
            base_payment = payments[task_id]
            combined_penalty = (agent_rank + 1) * (
                task_rank + 1
            )  # Convert to 1-indexed for penalty
            skill_multiplier = agent_skill / 10000  # Normalize skill
            adjusted_payment = (base_payment / combined_penalty) * skill_multiplier

            adjusted_payments[agent_id] = adjusted_payment
            agent_ranks[agent_id] = (
                agent_rank  # How much task wanted this agent (0-indexed)
            )
            task_ranks[agent_id] = (
                task_rank  # How much agent wanted this task (0-indexed)
            )

        return {
            "matches": matches,
            "base_payments": payments,
            "adjusted_payments": adjusted_payments,
            "agent_ranks": agent_ranks,  # Each agent's rank in their matched task (0-indexed)
            "task_ranks": task_ranks,  # Each agent's preference rank for their matched task (0-indexed)
            "matched_agents": len(matches),
            "unmatched_agents": len(agents) - len(matches),
            "task_preferences": task_preferences,  # Keep original task preferences intact
        }


# Example Agent placeholder (for testing)
class MockAgent:
    def __init__(self, agent_id: int, n_tasks: int):
        self.agent_id = agent_id
        self.skills = np.random.exponential(1.0, n_tasks)
        self.preferences = np.random.permutation(n_tasks)

    def get_skills(self) -> np.ndarray:
        return self.skills

    def get_preferences(self) -> np.ndarray:
        return self.preferences


# Quick test
if __name__ == "__main__":
    # Create market and agents
    market = LaborMarket(n_tasks=3)
    agents = [MockAgent(i, n_tasks=3) for i in range(5)]

    # Run simulation
    result = market.simulate_timestep(agents)
    print("Matches:", result["matches"])
    print("Payments:", result["payments"])
    print(
        f"Matched: {result['matched_agents']}, Unmatched: {result['unmatched_agents']}"
    )


# %%
# %%
def skill_weighted_ranking(skills: np.ndarray) -> np.ndarray:
    """Efficient skill-weighted ranking without repeated sampling"""

    # return np.argsort(-skills)
    uniform_random = np.random.uniform(0, 1, len(skills))
    # uniform_random = 1

    scores = uniform_random ** (1.0 / skills)

    print(1 / skills, scores)

    return np.argsort(-scores)  # Descending order (best first)


skills = np.array([1000, 1, 1, 1])
skill_weighted_ranking(skills)
# %%
1 ** (1 / skills)
# %%
