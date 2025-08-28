# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List


def skill_weighted_ranking(skills: np.ndarray, t=0.1) -> np.ndarray:
    # Ensure temperature is not zero to avoid division errors.
    # A small epsilon is used for numerical stability if t is very close to 0.
    t = max(t, 1e-9)

    # Gumbel noise is independent of temperature
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(skills))))

    # Original log probabilities (logits)
    log_probs = np.log(skills)

    # *** THIS IS THE KEY CHANGE ***
    # Scale the logits by temperature BEFORE adding the noise
    perturbed_skills = (log_probs / t) + gumbel_noise

    # Return indices sorted by the new perturbed scores (descending)
    return np.argsort(-perturbed_skills)


def visualize_ranking_distribution(skills: np.ndarray, n_samples: int = 10000, figsize: tuple = (12, 8), t=0.1) -> None:
    """
    Visualize the distribution of rankings from skill_weighted_ranking.

    Args:
        skills: Array of skill values
        n_samples: Number of times to sample the ranking
        figsize: Figure size for the plot
    """
    # Sample rankings n times
    ranking_counts = defaultdict(lambda: defaultdict(int))

    for _ in range(n_samples):
        ranking = skill_weighted_ranking(skills, t=t)
        for position, index in enumerate(ranking):
            ranking_counts[position][index] += 1

    # Convert to probability distributions
    n_items = len(skills)
    positions = list(range(n_items))

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: For each ranking position, show distribution over indices
    colors = plt.cm.Set3(np.linspace(0, 1, n_items))
    bar_width = 0.8 / n_items

    for idx in range(n_items):
        probs = [ranking_counts[pos][idx] / n_samples for pos in positions]
        x_pos = np.arange(len(positions)) + idx * bar_width
        ax1.bar(x_pos, probs, bar_width, label=f"A{idx} (s={skills[idx]:.2f})", color=colors[idx], alpha=0.7)

    ax1.set_xlabel("Ranking Position (1=best, higher=worse)")
    ax1.set_ylabel("Probability")
    ax1.set_title("Distribution of Indices at Each Ranking Position")
    ax1.set_xticks(np.arange(len(positions)) + bar_width * (n_items - 1) / 2)
    ax1.set_xticklabels([f"{i+1}" for i in positions])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heatmap showing the probability matrix
    prob_matrix = np.zeros((n_items, n_items))
    for pos in range(n_items):
        for idx in range(n_items):
            prob_matrix[idx, pos] = ranking_counts[pos][idx] / n_samples

    im = ax2.imshow(prob_matrix, cmap="YlOrRd", aspect="auto")
    ax2.set_xlabel("Ranking Position")
    ax2.set_ylabel("Index")
    ax2.set_title("Probability Heatmap\n(darker = higher probability)")
    ax2.set_xticks(range(n_items))
    ax2.set_xticklabels([f"{i+1}" for i in range(n_items)])
    ax2.set_yticks(range(n_items))
    ax2.set_yticklabels([f"A{i} (s={skills[i]:.2f})" for i in range(n_items)])

    # Add text annotations to heatmap
    for i in range(n_items):
        for j in range(n_items):
            text = ax2.text(j, i, f"{prob_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()

    # # Print summary statistics
    # print(f"\nSummary for skills {skills}:")
    # print("="*50)
    # for pos in range(n_items):
    #     print(f"Rank {pos+1} position:")
    #     for idx in range(n_items):
    #         prob = ranking_counts[pos][idx] / n_samples
    #         print(f"  Index {idx} (skill={skills[idx]:.2f}): {prob:.3f}")
    #     print()


# %%
normalize = lambda x: x / (sum(x) + 1e-9)


def harmonic_mean(reputation, bid, beta=1): 

    r_safe = reputation + 1e-9
    p_safe = (1.1 - bid) + 1e-9

    return (1 + beta ** 2) * (r_safe * p_safe) / ((beta ** 2) * p_safe + r_safe)
    
agent_bid = np.array([1, 0.3, 0.8, 0.7] + [1] * 6)  # float representing the % of budget
agent_reputation = np.array([1] + list(np.linspace(0.5, 0.1, 9)))

print(agent_reputation)

alpha = 0.9

# --- Model Definitions ---
def exp_score(score, steepness=3):
    # return score
    return 1 - np.exp(-steepness * (score))

def linear_model(reputation, bid, alpha=0.5):
    """
    Calculates score using a weighted linear model.
    alpha: weight for reputation (0 to 1).
    """
    price_score = 1.1 - bid

    price_score = exp_score(price_score, steepness=3)
    reputation = exp_score(reputation, steepness=1.5)

    print(reputation)
    print(price_score)

    return alpha * reputation + (1 - alpha) * price_score

def multiplicative_model(reputation, bid, w_rep=1.0, w_price=1.0):
    """
    Calculates score using a multiplicative model.
    w_rep, w_price: weights (exponents) for reputation and price.
    """
    # Add a small epsilon to avoid issues with log(0) or 0^0 if we were to take logs
    reputation = np.maximum(reputation, 1e-9)
    price_score = np.maximum(1.1 - bid, 1e-9)

    price_score = exp_score(price_score, steepness=5)
    reputation = exp_score(reputation, steepness=2)
    
    return (reputation ** w_rep) * (price_score ** w_price)

agent_fitness = linear_model(agent_reputation, agent_bid, alpha=0.8)

print(agent_fitness)
# print(np.array(agent_fitness).round(2))
visualize_ranking_distribution(agent_fitness, n_samples=100000, t=0.1)

# # Test with another example
# skills_example2 = np.array([5, 3, 7, 2])
# visualize_ranking_distribution(skills_example2, n_samples=10000)

# %%
from ssa.market import ExperimentLog

experiment = ExperimentLog.load("logs/oracle_t_0.2.log")

# %%
i = -1
history = experiment.history[i]
agent_scores = np.array(list(history.agent_reputation["cip_a"]))
print(agent_scores.round(2))

history.task_rewards


# %%
visualize_ranking_distribution(agent_scores, n_samples=10000, t=1)

# %%
plt.hist(-np.log(-np.log(np.random.uniform(0, 0.01, 10000))), bins=100)

# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Model Definitions ---
def exp_score(score, steepness=5):
    # return score
    return 1 - np.exp(-steepness * (score))

def linear_model(reputation, bid, alpha=0.5):
    """
    Calculates score using a weighted linear model.
    alpha: weight for reputation (0 to 1).
    """
    price_score = 1 - bid

    price_score = exp_score(price_score, steepness=1)
    reputation = exp_score(reputation, steepness=1)
    return alpha * reputation + (1 - alpha) * price_score

def multiplicative_model(reputation, bid, w_rep=1.0, w_price=1.0):
    """
    Calculates score using a multiplicative model.
    w_rep, w_price: weights (exponents) for reputation and price.
    """
    # Add a small epsilon to avoid issues with log(0) or 0^0 if we were to take logs
    reputation = np.maximum(reputation, 1e-9)
    price_score = np.maximum(1 - bid, 1e-9)

    price_score = exp_score(price_score)
    
    return (reputation ** w_rep) * (price_score ** w_price)

def harmonic_mean(reputation, bid, beta=1): 

    r_safe = reputation + 1e-9
    p_safe = exp_score(1-bid) + 1e-9

    return (1 + beta ** 2) * (r_safe * p_safe) / ((beta ** 2) * p_safe + r_safe)
    
# --- Visualization Setup ---

# Create a grid of possible reputation and bid values
reputation_vals = np.linspace(0, 1, 100)
bid_vals = np.linspace(1, 0, 100)
R, B = np.meshgrid(reputation_vals, bid_vals)

# --- Generate Plots ---

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Agent Ranking Score Landscapes: Comparing Models and Weights', fontsize=16)

# Define different preference scenarios
scenarios = {
    "Price Focused": {"alpha": 0.2, "w_rep": 0.5, "w_price": 1.5, "beta": 0.5},
    "Balanced":    {"alpha": 0.5, "w_rep": 1.0, "w_price": 1.0, "beta": 1},
    "Reputation Focused": {"alpha": 0.8, "w_rep": 2.0, "w_price": 0.5, "beta": 2},
}

# Plot for Linear Model
for i, (title, params) in enumerate(scenarios.items()):
    ax = axes[0, i]
    score = linear_model(R, B, alpha=params["alpha"])
    contour = ax.contourf(B, R, score, levels=20, cmap='nipy_spectral')
    ax.set_title(f"Linear Model\n{title} (alpha={params['alpha']})", fontsize=16)
    # ax.set_xlabel("Agent Bid (Normalized)")
    if i == 0: 
        ax.set_ylabel("Reputation (Normalized, higher = better)", fontsize=16)
    if i == 2:
        fig.colorbar(contour, ax=ax, label="Score")
        
    
# Plot for Multiplicative Model
for i, (title, params) in enumerate(scenarios.items()):
    ax = axes[1, i]
    score = multiplicative_model(R, B, w_rep=params["w_rep"], w_price=params["w_price"])
    contour = ax.contourf(B, R, score, levels=20, cmap='nipy_spectral')
    ax.set_title(f"Polynomial Model\n{title} (w_rep={params['w_rep']}, w_price={params['w_price']})", fontsize=16)
    # ax.set_xlabel("Agent Bid (Normalized)")
    if i == 0: 
        ax.set_ylabel("Reputation (Normalized, higher = better)", fontsize=16)
    
    if i == 2:
        fig.colorbar(contour, ax=ax, label="Score")
        
    
# Plot for Multiplicative Model
for i, (title, params) in enumerate(scenarios.items()):
    ax = axes[2, i]
    score = harmonic_mean(R, B, beta = params['beta'])
    contour = ax.contourf(B, R, score, levels=20, cmap='nipy_spectral')
    ax.set_title(f"Harmonic Mean\n{title} (beta={params['beta']})", fontsize=16)
    ax.set_xlabel("Agent Bid (Normalized, 1.0=100% of price)", fontsize=16)
    if i == 0: 
        ax.set_ylabel("Reputation (Normalized, higher = better)", fontsize=16)
    if i == 2:
        fig.colorbar(contour, ax=ax, label="Score")
        
    
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# %%

# %%
steepness = 1e-9


plt.plot(np.exp(steepness * np.linspace(0, 1, 10)))
# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Model Definitions ---

def power_transform(value, steepness=1.0):
    """
    Applies a non-linear power transformation to a score.
    - steepness > 1: Emphasizes high values (diminishing returns for low values).
    - steepness = 1: Linear, no change.
    - steepness < 1: Emphasizes low values (boosts them).
    """
    # Ensure value is non-negative to avoid issues with fractional exponents
    return 1 - np.exp(-steepness * (value))


def non_linear_linear_model(reputation, bid, alpha=0.8, rep_steepness=1.0, price_steepness=1.0):
    """
    Calculates score using a linear model on non-linearly transformed inputs.
    - alpha: Final weight for reputation (0 to 1).
    - rep_steepness: The steepness of the reputation utility curve.
    - price_steepness: The steepness of the price utility curve.
    """
    price_score = 1 - bid

    # Apply the non-linear transformation (utility function) to each component
    transformed_reputation = power_transform(reputation, steepness=rep_steepness)
    transformed_price = power_transform(price_score, steepness=price_steepness)
    
    # Return the simple weighted average of the *transformed* scores
    return alpha * transformed_reputation + (1 - alpha) * transformed_price


# --- Visualization Setup ---

# Create a grid of possible reputation and bid values
reputation_vals = np.linspace(0, 1, 100)
bid_vals = np.linspace(1, 0, 100)
R, B = np.meshgrid(reputation_vals, bid_vals)

# --- Generate Plots ---

# Define the parameters for the visualization
fixed_alpha = 0.8
rep_steepness_vals = [1.0, 1.5, 2]
price_steepness_vals = [1.0, 2, 3]

fig, axes = plt.subplots(len(rep_steepness_vals), len(price_steepness_vals), 
                         figsize=(18, 18), sharex=True, sharey=True)

fig.suptitle(f'Score Landscapes for Non-Linear Linear Model (fixed α = {fixed_alpha})', fontsize=20)

# Nested loop to create the 3x3 grid
for i, k_rep in enumerate(rep_steepness_vals):
    for j, s_p in enumerate(price_steepness_vals):
        ax = axes[i, j]
        
        # Calculate the score for the current combination of steepness values
        score = non_linear_linear_model(R, B, 
                                        alpha=fixed_alpha, 
                                        rep_steepness=k_rep, 
                                        price_steepness=s_p)
        
        # Plot the contour
        contour = ax.contourf(B, R, score, levels=20, cmap='nipy_spectral')
        
        # Set titles for each subplot
        ax.set_title(f"Rep Steepness = {k_rep}\nPrice Steepness = {s_p}", fontsize=14)
        
        # Add a colorbar to the rightmost plots
        if j == len(price_steepness_vals) - 1:
            fig.colorbar(contour, ax=ax, label="Score")

# Set shared axis labels
for i in range(len(rep_steepness_vals)):
    axes[i, 0].set_ylabel("Agent Reputation", fontsize=16)
for j in range(len(price_steepness_vals)):
    axes[len(rep_steepness_vals)-1, j].set_xlabel("Agent Bid", fontsize=16)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# %%
agent_reputation = np.linspace(0, 1, 100)
k_rep = 10
c_rep = 0.5
reputation_score = 1 / (1 + np.exp(-k_rep * (agent_reputation - c_rep))) / (1 / (1 + np.exp(-k_rep * (1 - c_rep))))

plt.plot(reputation_score)
reputation_score.max()
# %%
import math

def calculate_utility(price_norm, rep_norm, k_rep=10, c_rep=0.5, beta_rep=0.8, beta_price=0.2, gamma=1):
    """
    Calculates the utility score for an agent's bid.

    Args:
        price_norm (float): Agent's bid / job budget (e.g., 1.1 for 110%).
        rep_norm (float): Agent's reputation, normalized 0-1.
        persona (dict): A dictionary with client preference parameters.
                        {'beta_price': 5.0, 'beta_rep': 1.0, 'gamma': 0.1,
                         'k_rep': 10, 'c_rep': 0.7}
    """
    # 1. Calculate Reputation Utility using a Sigmoid function
    # T_reputation(r)
    rep_utility = 1 / (1 + np.exp(-k_rep * (rep_norm - c_rep)))

    # 2. Calculate the effective price sensitivity, modulated by reputation
    # This is the interaction effect
    effective_price_sensitivity = beta_price * (1 - gamma * rep_utility)

    # 3. Calculate the total utility
    # Utility = β_rep * T_rep(r) - (effective price sensitivity) * p
    # We use price_norm directly as the "disutility" part.
    total_utility = beta_rep * rep_utility - effective_price_sensitivity * price_norm

    return total_utility


# Create a grid of possible reputation and bid values
reputation_vals = np.linspace(0, 1, 100)
bid_vals = np.linspace(0, 1, 100)
R, B = np.meshgrid(reputation_vals, bid_vals)

# --- Generate Plots ---

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

score = calculate_utility(B, R, k_rep=10, c_rep=0.7, beta_rep=1, beta_price=1, gamma=0.8)
print(score.max())
contour = ax.contourf(B, R, score, levels=500, cmap='nipy_spectral')
# ax.set_title(f"Linear Model\n{title} (alpha={params['alpha']})", fontsize=16)
# ax.set_xlabel("Agent Bid (Normalized)")

ax.set_ylabel("Reputation (Normalized, higher = better)")
ax.set_xlabel("Price (Normalized, lower = better)")

fig.colorbar(contour, ax=ax, label="Score")
    
# %%
