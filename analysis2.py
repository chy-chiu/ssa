# %%
import numpy as np
from typing import List, Dict, Optional, Tuple, Set, Any
import pandas as pd
from pydantic import BaseModel
from ssa.tasks.task import TaskRunner, TaskBase, ProxyAgent, ProxyTask
from copy import deepcopy
from ssa.agents import (
    AgentHistory,
    AgentActionResponse,
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
from ssa.market import RoundData, ExperimentLog

import matplotlib.pyplot as plt
import os

from collections import defaultdict

import numpy as np

def compute_trace_means(traces):
    time_values = defaultdict(list)
    [time_values[t].append(v) for trace in traces for t, v in trace if v < 2]
    return np.array([(t, float(np.mean(vs))) for t, vs in sorted(time_values.items())])

def interp_trace(trace):
    times, values = zip(*trace)
    return np.interp(np.arange(100), times, values)

# %%
## Ablation Experiments

# %% 
exp_logs = []

for f in os.listdir('logs/ablation'):
    exp_log = ExperimentLog.load(f'logs/ablation/{f}')

    exp_logs.append(exp_log)
# %%
rewards = []
for exp_log in exp_logs:
    # rewards.append(exp_log.agent_reward_history[:8, :].reshape((8, 10, -1)).sum(axis=1))

    # rewards.append(exp_log.agent_total_rewards[-1])

    # reward = exp_log.agent_total_rewards[-1]

    for i in range(0, len(exp_log.agent_total_rewards), 10):
        i += 9
        reward = exp_log.agent_total_rewards[i]
    
        if len(reward) == 8:
            rewards.append(reward)
        else:
            rewards.append(np.array(reward)[[0, 1, 2, 3, 4, 5, 6, 8]])
# %%
plt.plot()

# %%
performance_data = np.array(rewards[:-1]).transpose((1, 2, 0)).reshape((8, -1))
performance_data.shape
# %%
for r in rewards:
    print(','.join([f"{agent_id}:{round(_r)}" for _r, agent_id in zip(r, exp_log.agent_ids)]))
# %%
performance_data = np.array(rewards).T
performance_data = performance_data / performance_data.sum(axis=0)
performance_data.shape

performance_data = performance_data[[0, 4,5,6,1,2,3,7]]
# %%
performance_data = performance_data.reshape((8, 10, 10))[:, -7:, -1]
performance_data.shape

# %%
# %%
all_rows = []
for f in os.listdir('logs/ablation/'):
    exp_log = ExperimentLog.load(f'logs/ablation/{f}')
    # plt.figure()
    # plt.plot(exp_log.agent_total_rewards, label=exp_log.agent_ids)
    # plt.legend()
    for a, r in zip(exp_log.agent_ids, exp_log.agent_total_rewards[-1]):
        all_rows.append((f, a, r))

# %%
df = pd.DataFrame(all_rows)
df.pivot(columns=1, index=0)
# %%
for f in os.listdir('logs'):
    if 'ablation' in f:
        print(f)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from itertools import combinations

# Setup your data structure
agent_configs = [
    (True, True, True),   # All abilities
    (True, True, False),  # A + B
    (True, False, True),  # A + C
    (False, True, True),  # B + C
    (True, False, False), # A only
    (False, True, False), # B only  
    (False, False, True), # C only
    (False, False, False),
]

# Example performance data (replace with your actual data)
# Shape: (7 agents, 10 runs)
# np.random.seed(42)
# performance_data = np.random.rand(7, 10) * 100  # Replace with your actual data

def analyze_ablation_study(configs, performance_data, ability_names=['M', 'C', 'P']):
    """
    Comprehensive ablation study analysis
    """
    # Convert to DataFrame for easier analysis
    results = []
    for i, config in enumerate(configs):
        for run in range(performance_data.shape[1]):
            results.append({
                'agent_id': i,
                'config': config,
                'M': config[0], 
                'C': config[1], 
                'P': config[2],
                'performance': performance_data[i, run],
                'run': run
            })
    
    df = pd.DataFrame(results)
    
    # 1. Main Effects Analysis
    print("=== MAIN EFFECTS ANALYSIS ===")
    main_effects = {}
    for ability in ability_names:
        present = df[df[ability] == True]['performance']
        absent = df[df[ability] == False]['performance']
        
        effect_size = present.mean() - absent.mean()
        p_value = stats.ttest_ind(present, absent)[1]
        
        main_effects[ability] = {
            'effect_size': effect_size,
            'p_value': p_value,
            'present_mean': present.mean(),
            'absent_mean': absent.mean(),
            'present_std': present.std(),
            'absent_std': absent.std()
        }
        
        print(f"{ability}: Effect = {effect_size:.3f}, p = {p_value:.4f}")
        print(f"  Present: {present.mean():.3f} ± {present.std():.3f}")
        print(f"  Absent:  {absent.mean():.3f} ± {absent.std():.3f}")
    
    # 2. Interaction Effects Analysis
    print("\n=== INTERACTION EFFECTS ANALYSIS ===")
    
    # Two-way interactions
    interactions = {}
    for ability1, ability2 in combinations(ability_names, 2):
        # Calculate interaction effect
        both_present = df[(df[ability1] == True) & (df[ability2] == True)]['performance'].mean()
        only_1 = df[(df[ability1] == True) & (df[ability2] == False)]['performance'].mean()
        only_2 = df[(df[ability1] == False) & (df[ability2] == True)]['performance'].mean()
        neither = df[(df[ability1] == False) & (df[ability2] == False)]['performance'].mean()
        
        # Interaction = (both - only_1) - (only_2 - neither)
        interaction_effect = (both_present - only_1) - (only_2 - neither)
        
        interactions[f"{ability1}×{ability2}"] = interaction_effect
        print(f"{ability1}×{ability2} interaction: {interaction_effect:.3f}")
    
    # # 3. Linear Regression Analysis
    # print("\n=== REGRESSION ANALYSIS ===")
    # X = df[ability_names].astype(int)
    # y = df['performance']
    
    # # Add interaction terms
    # X_with_interactions = X.copy()
    # for ability1, ability2 in combinations(ability_names, 2):
    #     X_with_interactions[f"{ability1}×{ability2}"] = X[ability1] * X[ability2]
    
    # # Three-way interaction
    # X_with_interactions['A×B×C'] = X['A'] * X['B'] * X['C']
    
    # model = LinearRegression()
    # model.fit(X_with_interactions, y)
    
    # print("Regression coefficients:")
    # for feature, coef in zip(X_with_interactions.columns, model.coef_):
    #     print(f"  {feature}: {coef:.3f}")
    # print(f"R² = {model.score(X_with_interactions, y):.3f}")
    
    # # 4. Contribution Analysis
    # print("\n=== CAPABILITY CONTRIBUTION ANALYSIS ===")
    
    # # Calculate relative importance
    # baseline_performance = 0  # No abilities case (not in your data, so assume 0)
    # full_performance = df[df['config'] == (True, True, True)]['performance'].mean()
    
    # contributions = {}
    # for ability in ability_names:
    #     only_ability = tuple(True if i == ability_names.index(ability) else False for i in range(3))
    #     if only_ability in configs:
    #         idx = configs.index(only_ability)
    #         solo_performance = performance_data[idx].mean()
    #         contribution = solo_performance - baseline_performance
    #         contributions[ability] = contribution / full_performance * 100  # As percentage
    #         print(f"{ability} solo contribution: {contribution:.3f} ({contributions[ability]:.1f}% of full)")
    
    return df, main_effects, interactions, model, contributions

def add_sig_bracket(ax, x1, x2, y, h, text, color='k', linewidth=1.5, fontsize=10):
    # Draw a square-bracket-like connector
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=linewidth, c=color, solid_capstyle='butt')
    ax.text((x1 + x2) / 2.0, y + h, text, ha='center', va='bottom', color=color, fontsize=fontsize)

# 5. Visualization
def plot_ablation_results(configs, performance_data, ability_names=['M', 'C', 'P']):
    """
    Create comprehensive visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance by configuration
    means = [performance_data[i].mean() for i in range(len(configs))]
    stds = [pd.Series(performance_data[i]).sem() for i in range(len(configs))]
    config_labels = [f"{'+'.join([ability_names[j] for j, x in enumerate(config) if x]) or 'Baseline'}" 
                    for config in configs]
    
    ax = axes[0]
    ax.bar(range(len(configs)), means, yerr=stds, capsize=5)
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Performance')
    ax.set_title('Performance by Ablation Configuration', fontsize=20)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(config_labels, rotation=45)
    ax.set_ylim(0, 0.2)
    
    # # Plot 2: Main effects
    df_analysis, main_effects, _, _, _ = analyze_ablation_study(configs, performance_data, ability_names)
    
    # abilities = list(main_effects.keys())
    # effects = [main_effects[ability]['effect_size'] for ability in abilities]
    # p_values = [main_effects[ability]['p_value'] for ability in abilities]
    
    # colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    # print(p_values)
    # bars = axes[0,1].bar(abilities, effects, color=colors)
    # axes[0,1].set_ylabel('Effect Size')
    # axes[0,1].set_title('Main Effects (Red = p < 0.05)')
    # axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot 3: Performance distribution by ability presence
    ax = axes[1]

    for i, ability in enumerate(ability_names):

        _df = df_analysis #[df_analysis['agent_id'].isin([0, 4, 5, 6, 7])]
        present = _df[_df[ability] == True]['performance']
        absent = _df[_df[ability] == False]['performance']
        
        ax.boxplot([absent, present], positions=[i*3, i*3+1], widths=0.6)
        # axes[1,0].text(i*3+0.5, max(present.max(), absent.max()) + 1, ability, ha='center')

    ax.set_ylabel('Performance')
    ax.set_title('Performance Distribution: Absent vs Present', fontsize=20)
    ax.set_xticks([0.5, 3.5, 6.5])

    y = 0.4
    h = 0.01  # bracket bottom
    add_sig_bracket(ax, 0, 1, y=y, h=h, text="***")
    add_sig_bracket(ax, 3, 4, y=y, h=h, text="*")

    
    ax.set_xticklabels(['Metacognition', 'Competitive Awareness', 'Planning'])
    
    # # Plot 4: Correlation matrix of abilities and performance
    # corr_data = df_analysis[ability_names + ['performance']].corr()
    # sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
    # axes[1,1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

# Run the analysis
df, main_effects, interactions, model, contributions = analyze_ablation_study(
    agent_configs, performance_data
)

plot_ablation_results(agent_configs, performance_data)

# 6. Statistical Significance Testing
def statistical_significance_test(configs, performance_data):
    """
    Comprehensive statistical testing
    """
    print("\n=== STATISTICAL SIGNIFICANCE TESTS ===")
    
    # ANOVA for overall differences
    groups = [performance_data[i] for i in range(len(configs))]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"Overall ANOVA: F = {f_stat:.3f}, p = {p_value:.4f}")
    
    # Post-hoc pairwise comparisons for key contrasts
    print("\nKey pairwise comparisons:")
    
    # Compare full model vs single abilities
    full_idx = configs.index((True, True, True))
    
    # for i in range(1, 8):
    #     t_stat, p_val = stats.ttest_ind(performance_data[full_idx], performance_data[i])
    #     ability_name = ['M', 'B', 'C', 'AB', 'AC', 'BC', 'None'][i-1]
    #     print(f"Full vs {ability_name}: t = {t_stat:.3f}, p = {p_val:.4f}")
    # print('\n')
    # for i in range(7):
    #     t_stat, p_val = stats.ttest_ind(performance_data[-1], performance_data[i])
    #     ability_name = ['Full', 'A', 'B', 'C', 'AB', 'AC', 'BC'][i]
    #     print(f"None vs {ability_name}: t = {t_stat:.3f}, p = {p_val:.4f}")


statistical_significance_test(agent_configs, performance_data)

# %%
performance_data.shape
# %%
partial = ExperimentLog.load('logs/moral/partial.log')
full = ExperimentLog.load('logs/moral/full.log')
# %%
price_skill = []
for experiment in [partial]:
    price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))
period = 10
n_agents = 4

non_price_skill = []
for experiment in [full]:
    non_price_skill.append(np.array([np.sum([a.action == 'train' for a in hx.agent_actions[:-2]]) for hx in experiment.history]))

prc = pd.Series(np.mean(price_skill, axis=0)/n_agents*100).ewm(span=period).mean().to_numpy()
noprc =  pd.Series(np.mean(non_price_skill, axis=0)/n_agents*100).ewm(span=period).mean().to_numpy()

plt.plot(prc, label='open')
plt.plot(noprc, label='sealed')

plt.fill_between(
    np.arange(100),prc, noprc, where=(prc > noprc), 
    interpolate=True, color="tab:blue", alpha=0.25, 
)

plt.fill_between(
    np.arange(100),prc, noprc, where=(noprc > prc), 
    interpolate=True, color="tab:orange", alpha=0.25,
)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Agents Training (%)", fontsize=20)
plt.xlabel("Timestep", fontsize=20)
plt.legend(fontsize=20)
plt.tight_layout()
# %%
plt.plot(np.array(full['SK-D']).T)
# %%
