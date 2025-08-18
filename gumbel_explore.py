# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List

def skill_weighted_ranking(skills: np.ndarray, t=0.1) -> np.ndarray:
    """Efficient skill-weighted ranking using Gumbel-Max trick."""
    # Add Gumbel noise to log-skills
    gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, len(skills)))) * t
    perturbed_skills = skills ** 2 + gumbel_noise

    # Return indices sorted by perturbed skills (descending)
    return np.argsort(-perturbed_skills)


def visualize_ranking_distribution(skills: np.ndarray, n_samples: int = 10000, 
                                 figsize: tuple = (12, 8), t=0.1) -> None:
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
        ax1.bar(x_pos, probs, bar_width, label=f'Index {idx} (skill={skills[idx]})', 
               color=colors[idx], alpha=0.7)
    
    ax1.set_xlabel('Ranking Position (0=best, higher=worse)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Distribution of Indices at Each Ranking Position')
    ax1.set_xticks(np.arange(len(positions)) + bar_width * (n_items-1) / 2)
    ax1.set_xticklabels([f'Rank {i+1}' for i in positions])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heatmap showing the probability matrix
    prob_matrix = np.zeros((n_items, n_items))
    for pos in range(n_items):
        for idx in range(n_items):
            prob_matrix[idx, pos] = ranking_counts[pos][idx] / n_samples
    
    im = ax2.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Ranking Position')
    ax2.set_ylabel('Index')
    ax2.set_title('Probability Heatmap\n(darker = higher probability)')
    ax2.set_xticks(range(n_items))
    ax2.set_xticklabels([f'Rank {i+1}' for i in range(n_items)])
    ax2.set_yticks(range(n_items))
    ax2.set_yticklabels([f'Idx {i} (skill={skills[i]})' for i in range(n_items)])
    
    # Add text annotations to heatmap
    for i in range(n_items):
        for j in range(n_items):
            text = ax2.text(j, i, f'{prob_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary for skills {skills}:")
    print("="*50)
    for pos in range(n_items):
        print(f"Rank {pos+1} position:")
        for idx in range(n_items):
            prob = ranking_counts[pos][idx] / n_samples
            print(f"  Index {idx} (skill={skills[idx]}): {prob:.3f}")
        print()

skills_example = np.array([0.1, 0.1, 0.2])
visualize_ranking_distribution(skills_example, n_samples=10000, t=0.05)

# # Test with another example
# skills_example2 = np.array([5, 3, 7, 2])
# visualize_ranking_distribution(skills_example2, n_samples=10000)

# %%
