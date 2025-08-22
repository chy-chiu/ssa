# %%
# 
# # Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulation Parameters ---
N_TIMESTEPS = 250
N_AGENTS = 10
EWMA_SPAN = 10 # How much history to consider for an agent's average

# Behavior parameters
IMPROVEMENT_RATE = 0.05  # How much skill improvers gain per turn
DECLINE_RATE = 0.05     # How much skill decliners lose per turn
RANDOM_WALK_STD = 0.1   # Volatility of random agents
PERFORMANCE_NOISE = 0.05 # Noise added to skill to get performance

REPUTATION_SENSITIVITY = 0.1 # Controls steepness of sigmoid

# For reproducibility
np.random.seed(42)

# --- 2. Agent Initialization ---
# Define agent roles
agent_types = ['Improver'] * 5 + ['Decliner'] * 3 + ['Random'] * 2
np.random.shuffle(agent_types) # Randomly assign roles to agent IDs

# Create a DataFrame to hold the current state of all agents
agents_df = pd.DataFrame({
    'type': agent_types,
    # Start all agents with a similar skill and reputation
    'skill': np.random.normal(loc=0.5, scale=0.05, size=N_AGENTS),
    'reputation': [0.5] * N_AGENTS,
    # Each agent's history is stored in a list
    'history': [[1] for _ in range(N_AGENTS)] 
})
agents_df['skill'] = agents_df['skill'].clip(0.01, 0.99)

# --- 3. Simulation Loop ---
history_log = []
performance_history = []
print("Running turn-based simulation with diverse agent behaviors...")

for t in range(N_TIMESTEPS):
    # a. Select one agent to perform a task at random
    acting_agent_id = np.random.choice(N_AGENTS)
    agent_type = agents_df.loc[acting_agent_id, 'type']
    
    # b. The agent performs a task. Performance = current skill + noise
    current_skill = agents_df.loc[acting_agent_id, 'skill']
    performance = np.clip(current_skill + np.random.normal(0, PERFORMANCE_NOISE), 0, 1)
    performance_history.append(performance)
    
    # c. Add performance to the agent's personal history
    agents_df.loc[acting_agent_id, 'history'].append(performance)
    agent_history = agents_df.loc[acting_agent_id, 'history']

    # d. Calculate the agent's new reputation based on the cohort history
    if len(performance_history) < 5:
        # Not enough history for EWMA, use a simple expanding mean
        historical_mean = np.mean(performance_history)
        historical_std = np.std(performance_history) if len(performance_history) > 1 else 1e-6
    else:
        # Use EWMA for a more responsive historical average
        history_series = pd.Series(performance_history)
        historical_mean = history_series.ewm(span=EWMA_SPAN).mean().iloc[-1]
        historical_std = history_series.rolling(window=EWMA_SPAN, min_periods=2).std().iloc[-1]

    # Sanitize std dev to prevent division by zero
    historical_std = max(historical_std, 1e-6)

    if agent_type == "Improver" and (t >= 180) and (t <= 190):
        performance = 0
    # Standardize current performance against the agent's own past
    standardized_performance = (performance - historical_mean) / historical_std
    
    # Update reputation using the sigmoid function
    
      # --- MODIFICATION START ---
    # 1. Calculate the purely relative reputation (surprise/momentum)
    relative_reputation = REPUTATION_SENSITIVITY * np.tanh(standardized_performance)
    curr_reputation = agents_df.loc[acting_agent_id, 'reputation']
    
    curr_reputation = np.clip(curr_reputation + relative_reputation, 0, 1)
    
    # 2. Gate this with the absolute performance to get the final score
    final_reputation = relative_reputation * performance
    
    # new_reputation = 1 / (1 + np.exp(-REPUTATION_SENSITIVITY * standardized_performance))
    agents_df.loc[acting_agent_id, 'reputation'] = curr_reputation

    # e. Update the agent's underlying skill based on its type
    if agent_type == 'Improver':
        agents_df.loc[acting_agent_id, 'skill'] += IMPROVEMENT_RATE
    elif agent_type == 'Decliner':
        agents_df.loc[acting_agent_id, 'skill'] -= DECLINE_RATE
    else: # Random
        agents_df.loc[acting_agent_id, 'skill'] += np.random.normal(0, RANDOM_WALK_STD)
            
    # Ensure skill stays within bounds [0, 1]
    agents_df['skill'] = agents_df['skill'].clip(0.01, 0.99)
    
    # f. Log the state of ALL agents at this timestep
    # for i in range(N_AGENTS):
    history_log.append({
        'timestep': t,
        'agent_id': acting_agent_id,
        'type': agents_df.loc[acting_agent_id, 'type'],
        'skill': current_skill,
        'performance': performance,
        'reputation': agents_df.loc[acting_agent_id, 'reputation']
    })

print("Simulation complete. Visualizing results...")

# --- 4. Data Analysis & Visualization ---
history_df = pd.DataFrame(history_log)

# Find one agent of each type to showcase
improver_id = history_df[history_df['type'] == 'Improver']['agent_id'].iloc[0]
decliner_id = history_df[history_df['type'] == 'Decliner']['agent_id'].iloc[0]
random_id = history_df[history_df['type'] == 'Random']['agent_id'].iloc[0]
agent_ids_to_plot = [improver_id, decliner_id, random_id]

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
fig.suptitle('Reputation System Tracking Different Agent Behaviors', fontsize=18)

plot_colors = {'skill': 'blue', 'reputation': 'green'}

for i, agent_id in enumerate(agent_ids_to_plot):
    agent_data = history_df[history_df['agent_id'] == agent_id]
    agent_type = agent_data['type'].iloc[0]
    ax = axes[i]
    
    ax.plot(agent_data['timestep'], agent_data['skill'], label='Hidden Skill (Ground Truth)', color=plot_colors['skill'], linewidth=2.5, alpha=0.8)
    ax.plot(agent_data['timestep'], agent_data['performance'], label='Performance', color='red', linewidth=1, linestyle='--')
    ax.plot(agent_data['timestep'], agent_data['reputation'], label='Calculated Reputation', color=plot_colors['reputation'], linewidth=2, linestyle='--')
    
    ax.set_title(f'Agent {agent_id} (Type: {agent_type})', fontsize=14)
    ax.set_ylabel('Score')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

axes[-1].set_xlabel('Timestep', fontsize=12)

# performance = 


axes[3].plot(pd.Series(performance_history))
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
-23
# %%
# %%
plt.plot(performance_history)
# %%
agent_data
# %%
plt.plot(agent_data['timestep'], agent_data['performance'])
# %%
len(performance_history)
# %%
history_df.type.unique()
# %%
