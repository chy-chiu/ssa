# %%
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from market import LabourMarket

class MockAgent:
    """Unified agent class for both static and RL agents"""
    def __init__(self, agent_id: int, skills: np.ndarray = None, preferences: List[int] = None):
        self.agent_id = agent_id
        self.skills = skills if skills is not None else np.array([1.0, 1.0, 1.0])
        self.preferences = preferences if preferences is not None else [0, 1, 2]
    
    def get_skills(self) -> np.ndarray:
        return self.skills
    
    def get_preferences(self) -> np.ndarray:
        return np.array(self.preferences)
    
    def upgrade_skill(self, task_id: int, base_amount: float = 0.5):
        """
        Upgrade skill for a specific task with convex growth that asymptotes to 1000
        Uses formula: new_skill = old_skill + base_amount * (1000 - old_skill) / 10
        """
        current_skill = self.skills[task_id]
        if current_skill < 10000:
            # Convex growth that slows down as skill approaches 1000
            growth = base_amount * (10000 - current_skill) / 10000
            self.skills[task_id] = min(current_skill + growth, 10000.0)
        
class LaborMarketEnv(gym.Env):
    """Gym environment for labor market RL with persistent agents and skill growth"""
    
    def __init__(self, market: LabourMarket, rl_agent_id: int = 0):
        super(LaborMarketEnv, self).__init__()
        
        self.market = market
        self.rl_agent_id = rl_agent_id
        self.n_tasks = market.n_tasks
        
        # Initialize all agents once and persist them
        self.agents = self._create_all_agents()
        self.rl_agent = self.agents[self.rl_agent_id]  # Reference to RL agent
        
        # Action space: all possible preference orderings
        self.action_space = spaces.Discrete(6)  
        self.task_orderings = [
            [0, 1, 2], [0, 2, 1], [1, 0, 2], 
            [1, 2, 0], [2, 0, 1], [2, 1, 0]
        ]
        
        # Observation space: only current payments + own skills + match history
        # RL agent should NOT see competitor skills
        obs_size = 3 + 3 + 3  # payments + own_skills + last_matches
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Tracking
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.step_count = 0
        self.total_steps = 0  # Track across all episodes
        self.last_matches = [-1, -1, -1]  # Last 3 matches (-1 = no match)
        
    def _create_all_agents(self) -> List[MockAgent]:
        """Create all agents including RL agent"""
        agents = []
        
        # Agent 0: RL agent (starts with equal skills)
        rl_agent = MockAgent(
            agent_id=0,
            skills=np.array([1.0, 1.0, 1.0]),
            preferences=[0, 1, 2]  # Will be overridden by actions
        )
        agents.append(rl_agent)
        
        # Agent 1: Expert in task 0, always prefers task 0
        agent1 = MockAgent(
            agent_id=1,
            skills=np.array([1.0, 1.0, 1.0]),
            preferences=[0, 1, 2]
        )
        agents.append(agent1)
        
        # Agent 2: Expert in task 1, always prefers task 1  
        agent2 = MockAgent(
            agent_id=2,
            skills=np.array([1.0, 1.0, 1.0]),
            preferences=[1, 0, 2]
        )
        agents.append(agent2)
        
        # Agent 3: Competes for task 2, averages in all tasks
        agent3 = MockAgent(
            agent_id=3,
            skills=np.array([1.0, 1.0, 1.0]),
            preferences=[2, 0, 1]
        )
        agents.append(agent3)
        
        return agents
    
    def reset(self, seed=None):
        """Reset episode (but agents and their skills persist!)"""
        if self.step_count > 0:
            self.episode_rewards.append(self.current_episode_reward)
        
        self.current_episode_reward = 0
        self.step_count = 0
        self.last_matches = [-1, -1, -1]
        
        # Generate initial payments
        self.current_payments = self.market.generate_tasks()
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation - only what RL agent should know"""
        # Convert last matches to encoding
        match_encoding = np.zeros(3)
        for i, match in enumerate(self.last_matches):
            if match >= 0:
                match_encoding[match] = 1.0
        
        obs = np.concatenate([
            self.current_payments,
            self.rl_agent.get_skills(),  # Only RL agent's own skills
            match_encoding
        ]).astype(np.float32)
        return obs
    
    def step(self, action):
        """Take a step in the environment"""
        # Update RL agent's preferences based on action
        rl_preferences = self.task_orderings[action]
        self.rl_agent.preferences = rl_preferences
        
        # Run market simulation with all agents
        result = self.market.simulate_timestep(self.agents)
        
        # Calculate reward using rank-adjusted payments
        reward = 0
        matched_task = -1
        agent_rank = None
        
        if self.rl_agent_id in result['adjusted_payments']:
            reward = result['adjusted_payments'][self.rl_agent_id]
            matched_task = result['matches'][self.rl_agent_id]
            agent_rank = result['agent_ranks'][self.rl_agent_id]
            
        # Upgrade skills for ALL agents who got matched (hidden from RL agent)
        for agent in self.agents:
            if agent.agent_id in result['matches']:
                completed_task = result['matches'][agent.agent_id]
                agent.upgrade_skill(completed_task, 0.5)
            else:
                agent.upgrade_skill(agent.preferences[0], 0.005)
        
        self.current_episode_reward += reward
        self.step_count += 1
        self.total_steps += 1
        
        # Update history
        self.last_matches.pop(0)
        self.last_matches.append(matched_task)
        
        # Generate new payments for next step
        self.current_payments = self.market.generate_tasks()
        
        # Episode continues
        done = self.step_count >= 1000  # End episode after 1000 steps
        
        info = {
            'matched_task': matched_task,
            'agent_rank': agent_rank,
            'base_payment': result['base_payments'][matched_task] if matched_task >= 0 else 0,
            'adjusted_payment': reward,
            'total_steps': self.total_steps,
            'all_matches': result['matches'],
            'all_ranks': result['agent_ranks'],
            'action_taken': action,
            'preferences': rl_preferences,
            # Hidden info for analysis (not observable by RL agent)
            '_rl_agent_skills': self.rl_agent.get_skills().copy(),
            '_all_agent_skills': [agent.get_skills().copy() for agent in self.agents]
        }
        
        return self._get_observation(), reward, done, False, info
    
    def get_skill_evolution(self):
        """Get current skill levels for analysis"""
        return {
            'rl_agent': self.rl_agent.get_skills().copy(),
            'competitors': [agent.get_skills().copy() for agent in self.agents[1:]]
        }

def train_rl_agent():
    """Train the RL agent with persistent skill growth"""
    # Create environment
    market = LabourMarket(n_tasks=3)
    env = LaborMarketEnv(market)
    
    # Create PPO model
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.02,
    )
    
    # Training tracking
    rewards_over_time = []
    task_preference_rates = []
    task_matching_rates = []
    skill_evolution = []  # Track how skills evolve
    action_distributions = []
    
    # Training loop with evaluation
    total_timesteps = 10000
    eval_freq = 1000
    
    for i in range(0, total_timesteps, eval_freq):
        print(f"\nTraining step {i}-{i+eval_freq}")
        
        # Record skills before training
        skills_snapshot = env.get_skill_evolution()
        skill_evolution.append(skills_snapshot)
        
        # Train
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        
        # Evaluate
        print("Evaluating...")
        episode_rewards = []
        task_preferences = [0, 0, 0]
        task_matches = [0, 0, 0]
        actions_taken = []
        
        for episode in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                actions_taken.append(action)
                
                # Track preferences and matches
                first_choice_task = env.task_orderings[action][0]
                task_preferences[first_choice_task] += 1
                
                if info['matched_task'] >= 0:
                    task_matches[info['matched_task']] += 1
                
                steps += 1
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        total_decisions = len(actions_taken)
        
        task_pref_rates = [count / total_decisions for count in task_preferences] if total_decisions > 0 else [0, 0, 0]
        task_match_rates = [count / total_decisions for count in task_matches] if total_decisions > 0 else [0, 0, 0]
        
        rewards_over_time.append(avg_reward)
        task_preference_rates.append(task_pref_rates)
        task_matching_rates.append(task_match_rates)
        action_distributions.append(np.bincount(actions_taken, minlength=6) / len(actions_taken))
        
        # Print results including skill evolution
        current_skills = env.get_skill_evolution()
        print(f"Avg Reward: {avg_reward:.2f}")
        print("RL Agent Skills:", [f"{s:.1f}" for s in current_skills['rl_agent']])
        print("Task Preference Rates:", [f"{r:.3f}" for r in task_pref_rates])
        print("Task Match Rates:", [f"{r:.3f}" for r in task_match_rates])
        
        # Show most common preferences
        most_common_action = np.argmax(np.bincount(actions_taken))
        most_common_prefs = env.task_orderings[most_common_action]
        print(f"Most common preference order: {[x+1 for x in most_common_prefs]}")
    
    return model, rewards_over_time, task_preference_rates, task_matching_rates, skill_evolution, action_distributions, env

def plot_results(rewards, task_pref_rates, task_match_rates, skill_evolution, action_dists):
    """Plot comprehensive training results including skill evolution"""
    fig = plt.figure(figsize=(20, 12))
    
    steps = np.arange(len(rewards)) * 10000
    
    # Convert list of arrays to numpy arrays for easier plotting
    pref_rates_array = np.array(task_pref_rates)  # Shape: (n_eval_points, 3)
    match_rates_array = np.array(task_match_rates)  # Shape: (n_eval_points, 3)
    
    # 1. Average reward over time
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(steps, rewards, 'b-', linewidth=2)
    plt.title('Average Reward Over Time', fontsize=14)
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward per Episode')
    plt.grid(True, alpha=0.3)
    
    # 2. Task preference rates over time
    ax2 = plt.subplot(2, 3, 2)
    colors = ['red', 'green', 'blue']
    for task in range(3):
        plt.plot(steps, pref_rates_array[:, task], color=colors[task], 
                linewidth=2, label=f'Task {task+1}')
    plt.title('Task Preference Rates Over Time', fontsize=14)
    plt.xlabel('Training Steps')
    plt.ylabel('Preference Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Task match rates over time
    ax3 = plt.subplot(2, 3, 3)
    for task in range(3):
        plt.plot(steps, match_rates_array[:, task], color=colors[task], 
                linewidth=2, linestyle='--', label=f'Task {task+1}')
    plt.title('Task Match Rates Over Time', fontsize=14)
    plt.xlabel('Training Steps')
    plt.ylabel('Match Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Action distribution evolution (heatmap)
    ax4 = plt.subplot(2, 3, 4)
    action_matrix = np.array(action_dists).T
    im = ax4.imshow(action_matrix, aspect='auto', cmap='Blues', origin='lower')
    ax4.set_title('Action Distribution Over Time', fontsize=14)
    ax4.set_xlabel('Training Steps (x10k)')
    ax4.set_ylabel('Action ID')
    ax4.set_yticks(range(6))
    
    # Add action labels
    orderings = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
    labels = [f"{i}: {[x+1 for x in ord]}" for i, ord in enumerate(orderings)]
    ax4.set_yticklabels(labels, fontsize=8)
    
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    # 5. Final preference vs match comparison
    ax5 = plt.subplot(2, 3, 5)
    final_prefs = pref_rates_array[-1] if len(pref_rates_array) > 0 else [1/3, 1/3, 1/3]
    final_matches = match_rates_array[-1] if len(match_rates_array) > 0 else [1/3, 1/3, 1/3]
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, final_prefs, width, label='Preference Rate', 
                   color=['red', 'green', 'blue'], alpha=0.7)
    bars2 = ax5.bar(x + width/2, final_matches, width, label='Match Rate',
                   color=['red', 'green', 'blue'], alpha=0.4)
    
    ax5.set_title('Final Preferences vs Matches', fontsize=14)
    ax5.set_xlabel('Task')
    ax5.set_ylabel('Rate')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Task 1', 'Task 2', 'Task 3'])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Skill Evolution Over Time
    ax6 = plt.subplot(2, 3, 6)
    
    if skill_evolution and len(skill_evolution) > 0:
        # Extract RL agent skills over time
        rl_skills_over_time = np.array([snapshot['rl_agent'] for snapshot in skill_evolution])
        
        # Plot RL agent skill evolution
        for task in range(3):
            plt.plot(steps[:len(rl_skills_over_time)], rl_skills_over_time[:, task], 
                    color=colors[task], linewidth=3, label=f'RL Agent Task {task+1}')
        
        # Plot competitor skills (only their specialized skills for clarity)
        if len(skill_evolution[0]['competitors']) >= 3:
            # Competitor 0: specialized in task 0
            comp0_skill0 = [snapshot['competitors'][0][0] for snapshot in skill_evolution]
            plt.plot(steps[:len(comp0_skill0)], comp0_skill0, 
                    color='red', linewidth=1, linestyle=':', alpha=0.7, label='Comp1 Task 1')
            
            # Competitor 1: specialized in task 1  
            comp1_skill1 = [snapshot['competitors'][1][1] for snapshot in skill_evolution]
            plt.plot(steps[:len(comp1_skill1)], comp1_skill1,
                    color='green', linewidth=1, linestyle=':', alpha=0.7, label='Comp2 Task 2')
            
            # Competitor 2: specialized in task 2
            comp2_skill2 = [snapshot['competitors'][2][2] for snapshot in skill_evolution]
            plt.plot(steps[:len(comp2_skill2)], comp2_skill2,
                    color='blue', linewidth=1, linestyle=':', alpha=0.7, label='Comp3 Task 3')
    
    ax6.set_title('Skill Evolution Over Time', fontsize=14)
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Skill Level (log scale)')
    ax6.set_yscale('log')  # Use log scale due to large skill differences
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Add horizontal line at 1000 (asymptote)
    ax6.axhline(y=1000, color='black', linestyle='--', alpha=0.5, label='Asymptote (1000)')
    
    plt.tight_layout()
    plt.show()
    
    # Print final analysis including skills
    print("\n=== FINAL ANALYSIS ===")
    print(f"Final average reward: {rewards[-1]:.2f}")
    
    if skill_evolution:
        final_skills = skill_evolution[-1]
        print(f"\nFinal RL Agent Skills: {[f'{s:.1f}' for s in final_skills['rl_agent']]}")
        print("Final Competitor Skills:")
        for i, comp_skills in enumerate(final_skills['competitors']):
            print(f"  Competitor {i+1}: {[f'{s:.1f}' for s in comp_skills]}")
    
    print("\nFinal task preference rates:")
    for task in range(3):
        print(f"  Task {task+1}: {final_prefs[task]:.3f}")
    print("\nFinal task match rates:")
    for task in range(3):
        print(f"  Task {task+1}: {final_matches[task]:.3f}")
    
    print("\nTask orderings:")
    final_dist = action_dists[-1] if action_dists else np.ones(6)/6
    for i, ordering in enumerate([[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]):
        task_names = [x+1 for x in ordering]
        prob = final_dist[i]
        marker = " ⭐" if ordering[0] == 2 else ""  # Highlight task-3-first
        print(f"  Action {i}: {task_names} - {prob:.3f}{marker}")


# %%
if __name__ == "__main__":
    print("Training RL agent in labor market...")
    print("Setup: 2 static agents (experts in tasks 1&2), 1 RL agent learning")
    print("Expected: RL agent should learn to prefer task 3 first\n")
    
    # Run training
    model, rewards_over_time, task_preference_rates, task_matching_rates, skill_evolution, action_distributions, env = train_rl_agent()
    
    # Plot results  
    plot_results(rewards_over_time, task_preference_rates, task_matching_rates, skill_evolution, action_distributions)
    
    # Save model
    model.save("labor_market_ppo")
    print("\nModel saved as 'labor_market_ppo.zip'")
    

# %%
action_dists
# %%
skill_evolution
# %%
i = 2
np.array([a['rl_agent'][i] for a in skill_evolution])
# %%
skill_evolution

# %%
