import numpy as np
from typing import List, Dict, Set, Tuple

def multi_galeshapley(
    agent_preferences: List[List[str]], 
    market_preference: Dict[str, List[int]],
    multi_limit: int
) -> Tuple[Dict[str, int], Set[int], List[str]]:
    """
    Multi-job Gale-Shapley matching with JOBS proposing to agents
    Each agent can accept at most multi_limit jobs
    This version is "job-optimal" - gives best possible outcome for jobs
    
    Handles incomplete coverage:
    - Jobs may not be in all agents' preference lists (agent rejects)
    - Agents may reference non-existent jobs (filtered out)
    - Jobs may reference non-existent agents (filtered out)
    
    Args:
        agent_preferences: List of length n_agents, each containing job ranking (as strings)
        market_preference: Dict mapping job_id (string) to agent ranking
        multi_limit: Maximum number of jobs each agent can hold
        
    Returns:
        job_matches: Dict {job_id: agent_id} of matches
        unmatched_agents: Set of agent indices with no matches  
        unmatched_jobs: List of job_ids with no matches
    """
    n_agents = len(agent_preferences)
    
    # Clean up market preferences - remove invalid agent indices and duplicates
    cleaned_market_preference = {}
    for job_id, agent_list in market_preference.items():
        # Filter out invalid agent indices and remove duplicates while preserving order
        valid_agents = []
        seen = set()
        for agent_id in agent_list:
            agent_id = int(agent_id)
            if 0 <= agent_id < n_agents and agent_id not in seen:
                valid_agents.append(agent_id)
                seen.add(agent_id)
        cleaned_market_preference[job_id] = valid_agents
    
    # Clean up agent preferences - remove jobs that don't exist in market
    cleaned_agent_preferences = []
    available_jobs = set(cleaned_market_preference.keys())
    
    for agent_id in range(n_agents):
        if agent_id < len(agent_preferences):
            # Filter out non-existent jobs while preserving order
            valid_jobs = [job_id for job_id in agent_preferences[agent_id] if job_id in available_jobs]
        else:
            valid_jobs = []
        cleaned_agent_preferences.append(valid_jobs)
    
    # Tracking structures
    job_next_proposal = {job_id: 0 for job_id in cleaned_market_preference.keys()}
    job_current_match = {}  # {job_id: agent_id}
    agent_current_jobs = {i: set() for i in range(n_agents)}  # {agent_id: set(job_ids)}
    
    # Create agent preference rankings for quick lookup: {agent_id: {job_id: rank}}
    agent_job_rank = {}
    for agent_id, job_list in enumerate(cleaned_agent_preferences):
        agent_job_rank[agent_id] = {job_id: rank for rank, job_id in enumerate(job_list)}
    
    # Queue of jobs that need to find agents (only jobs with valid agent preferences)
    unmatched_jobs = [job_id for job_id, agent_list in cleaned_market_preference.items() if len(agent_list) > 0]
    
    while unmatched_jobs:
        # Randomly select which job to process next
        job_idx = np.random.randint(len(unmatched_jobs))
        job_id = unmatched_jobs.pop(job_idx)
        
        # Check if job has exhausted all preferred agents
        if job_next_proposal[job_id] >= len(cleaned_market_preference[job_id]):
            continue  # Job remains unmatched
            
        # Job proposes to its next preferred agent
        agent_id = cleaned_market_preference[job_id][job_next_proposal[job_id]]
        job_next_proposal[job_id] += 1
        
        # Check if agent has this job in their preference list
        if job_id not in agent_job_rank[agent_id]:
            # Agent doesn't have this job in preferences - automatic rejection
            unmatched_jobs.append(job_id)
            continue
        
        # Agent considers the job offer
        agent_accepts = False
        job_to_drop = None
        
        if len(agent_current_jobs[agent_id]) < multi_limit:
            # Agent has capacity, accepts the job
            agent_accepts = True
            
        else:
            # Agent is at capacity, compare with worst current job
            current_jobs = list(agent_current_jobs[agent_id])
            
            # Find agent's worst current job (highest rank = least preferred)
            worst_rank = -1
            worst_job = None
            for curr_job in current_jobs:
                if curr_job in agent_job_rank[agent_id]:  # Safety check
                    curr_rank = agent_job_rank[agent_id][curr_job]
                    if curr_rank > worst_rank:
                        worst_rank = curr_rank
                        worst_job = curr_job
            
            # Compare new job with worst current job
            if worst_job is not None:
                new_job_rank = agent_job_rank[agent_id][job_id]
                if new_job_rank < worst_rank:  # New job is better (lower rank = higher preference)
                    agent_accepts = True
                    job_to_drop = worst_job
        
        if agent_accepts:
            # Agent accepts the job
            if job_to_drop:
                # Drop worst job first
                agent_current_jobs[agent_id].remove(job_to_drop)
                del job_current_match[job_to_drop]
                unmatched_jobs.append(job_to_drop)  # Dropped job needs to find new agent
                
            # Assign new job to agent
            job_current_match[job_id] = agent_id
            agent_current_jobs[agent_id].add(job_id)
            
        else:
            # Agent rejects, job continues searching
            unmatched_jobs.append(job_id)
    
    # Calculate final unmatched sets
    unmatched_agents = set()
    for agent_id in range(n_agents):
        if len(agent_current_jobs[agent_id]) == 0:
            unmatched_agents.add(agent_id)
            
    final_unmatched_jobs = []
    for job_id in cleaned_market_preference.keys():
        if job_id not in job_current_match:
            final_unmatched_jobs.append(job_id)
    
    return job_current_match, unmatched_agents, final_unmatched_jobs


def test_match_jobs_multi():
    """
    Test case for multi-job matching with incomplete coverage
    
    Setup: 3 agents, 5 jobs, limit of 2 jobs per agent
    Some agents don't want all jobs, some jobs don't want all agents
    """
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Agent preferences - not all agents want all jobs
    agent_preferences = [
        ['J1', 'J2', 'J4'],          # Agent 0: only wants J1, J2, J4 (missing J3, J5)
        ['J2', 'J1', 'J3', 'J5'],    # Agent 1: wants J2, J1, J3, J5 (missing J4)  
        ['J3', 'J4', 'J1', 'J6']     # Agent 2: wants J3, J4, J1, and non-existent J6
    ]
    
    # Market preferences - not all jobs want all agents, some have invalid references
    market_preference = {
        'J1': [0, 1, 2],       # J1 wants all agents
        'J2': [1, 0],          # J2 only wants agents 1 and 0 (not agent 2)
        'J3': [2, 0],          # J3 only wants agents 2 and 0 (not agent 1)
        'J4': [0, 2, 1],       # J4 wants all agents
        'J5': [1, 2, 5],       # J5 wants agents 1, 2, and invalid agent 5
    }
    
    multi_limit = 2
    
    print("=== Initial Setup ===")
    print("Agent preferences:")
    for i, prefs in enumerate(agent_preferences):
        print(f"  Agent {i}: {prefs}")
    print("\nMarket preferences:")
    for job, prefs in market_preference.items():
        print(f"  {job}: {prefs}")
    
    job_matches, unmatched_agents, unmatched_jobs = multi_galeshapley(
        agent_preferences, market_preference, multi_limit
    )
    
    print("\n=== Multi-Job Matching Results ===")
    print(f"Job matches: {job_matches}")
    print(f"Unmatched agents: {unmatched_agents}")
    print(f"Unmatched jobs: {unmatched_jobs}")
    
    # Count jobs per agent
    agent_job_count = {0: 0, 1: 0, 2: 0}
    for job_id, agent_id in job_matches.items():
        agent_job_count[agent_id] += 1
    
    print(f"Jobs per agent: {agent_job_count}")
    
    # Show detailed agent assignments with preferences
    print("\n=== Agent Assignments (with preference ranks) ===")
    for agent_id in range(len(agent_preferences)):
        assigned_jobs = [job_id for job_id, assigned_agent in job_matches.items() if assigned_agent == agent_id]
        if assigned_jobs:
            # Show ranks only for jobs that exist in agent's cleaned preferences
            available_jobs = set(market_preference.keys())
            cleaned_prefs = [job for job in agent_preferences[agent_id] if job in available_jobs]
            
            job_ranks = []
            for job in assigned_jobs:
                if job in cleaned_prefs:
                    job_ranks.append(cleaned_prefs.index(job))
                else:
                    job_ranks.append("N/A")
            
            print(f"  Agent {agent_id}: {assigned_jobs} (preference ranks: {job_ranks})")
            print(f"    Agent's cleaned preferences: {cleaned_prefs}")
        else:
            print(f"  Agent {agent_id}: No jobs assigned")
    
    # Verify constraints
    print("\n=== Constraint Verification ===")
    
    # 1. No agent exceeds multi_limit
    for agent_id, count in agent_job_count.items():
        assert count <= multi_limit, f"Agent {agent_id} has {count} jobs, exceeds limit {multi_limit}"
    print("✓ All agents respect job limit")
    
    # 2. Each job assigned to at most one agent
    assigned_jobs = set(job_matches.keys())
    assert len(assigned_jobs) == len(job_matches), "Duplicate job assignments detected"
    print("✓ Each job assigned to exactly one agent")
    
    # 3. All matched jobs are mutually acceptable
    for job_id, agent_id in job_matches.items():
        # Check if agent has job in their cleaned preferences
        available_jobs = set(market_preference.keys())
        agent_cleaned_prefs = [job for job in agent_preferences[agent_id] if job in available_jobs]
        assert job_id in agent_cleaned_prefs, f"Agent {agent_id} matched to unwanted job {job_id}"
        
        # Check if job has agent in their cleaned preferences
        valid_agents = [a for a in market_preference[job_id] if 0 <= a < len(agent_preferences)]
        assert agent_id in valid_agents, f"Job {job_id} matched to unwanted agent {agent_id}"
    
    print("✓ All matches are mutually acceptable")
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_match_jobs_multi()
