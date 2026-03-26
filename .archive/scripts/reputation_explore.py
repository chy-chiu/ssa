# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import defaultdict, deque

class DynamicCommunityReputationSystem:
    def __init__(self, n_agents, window_size=20, community_window_size=30, 
                 longevity_factor=0.9, prior_alpha=1.0, prior_beta=1.0):
        """
        Dynamic community-based reputation system.
        
        Args:
            n_agents: Number of agents
            window_size: Size of sliding window for individual reputation
            community_window_size: Size of sliding window for community base rate
            longevity_factor: λ for exponential aging
        """
        self.n_agents = n_agents
        self.window_size = window_size
        self.community_window_size = community_window_size
        self.longevity_factor = longevity_factor
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Individual agent sliding windows
        self.rating_windows = [deque(maxlen=window_size) for _ in range(n_agents)]
        
        # Community sliding window for base rate calculation
        self.community_performance_window = deque(maxlen=community_window_size)
        
        # Track history
        self.complete_history = defaultdict(list)
        self.community_history = []
        self.current_time = 0
        
    def sigmoid_skill_evolution(self, agent_id, t, initial_skill=0.2, final_skill=0.9, 
                               midpoint=40, steepness=0.08):
        """Monotonically increasing sigmoid skill curve."""
        # np.random.seed(agent_id + 1000)
        agent_midpoint = midpoint # + np.random.normal(0, 5)
        agent_final = final_skill # + np.random.normal(0, 0.05)
        agent_final = np.clip(agent_final, 0.1, 0.95)
        
        skill = initial_skill + (agent_final - initial_skill) / (
            1 + np.exp(-steepness * (t - agent_midpoint))
        )
        return np.clip(skill, 0.05, 0.95)
    
    def get_current_community_base_rate(self):
        """
        Calculate dynamic community base rate from sliding window of all performances.
        """
        if not self.community_performance_window:
            return 0.5  # Default neutral base rate
        
        return sum(self.community_performance_window) / len(self.community_performance_window)
    
    def get_individual_reputation_sliding_window(self, agent_id):
        """
        Individual reputation using sliding window with dynamic community base rate as prior.
        """
        if not self.rating_windows[agent_id]:
            return self.get_current_community_base_rate()
        
        # Get current community base rate as Bayesian prior
        community_base_rate = self.get_current_community_base_rate()
        
        # Sum individual ratings in window
        total_performance = sum(rating['performance'] for rating in self.rating_windows[agent_id])
        n_ratings = len(self.rating_windows[agent_id])
        
        # Bayesian update with community base rate as prior
        # Prior: α = W * community_rate, β = W * (1 - community_rate)
        W = 2.0  # Prior weight
        prior_alpha = W * community_base_rate
        prior_beta = W * (1 - community_base_rate)
        
        posterior_alpha = prior_alpha + total_performance
        posterior_beta = prior_beta + (n_ratings - total_performance)
        
        return posterior_alpha / (posterior_alpha + posterior_beta)
    
    def get_individual_reputation_exponential_aging(self, agent_id):
        """
        Individual reputation using exponential aging with dynamic community base rate.
        """
        if not self.rating_windows[agent_id]:
            return self.get_current_community_base_rate()
        
        community_base_rate = self.get_current_community_base_rate()
        
        # Apply exponential weights to individual ratings
        weighted_sum = 0
        weight_sum = 0
        
        for rating in self.rating_windows[agent_id]:
            age = self.current_time - rating['timestamp']
            weight = self.longevity_factor ** age
            weighted_sum += weight * rating['performance']
            weight_sum += weight
        
        if weight_sum == 0:
            return community_base_rate
        
        # Bayesian update with community-informed prior
        W = 2.0
        prior_alpha = W * community_base_rate
        prior_beta = W * (1 - community_base_rate)
        
        effective_successes = weighted_sum
        effective_trials = weight_sum
        effective_failures = effective_trials - effective_successes
        
        posterior_alpha = prior_alpha + effective_successes
        posterior_beta = prior_beta + effective_failures
        
        return posterior_alpha / (posterior_alpha + posterior_beta)
    
    def update_reputation(self, agent_id, true_skill, noise_std=0.1):
        """
        Generate performance and update individual + community reputation.
        """
        # Generate performance with noise around true skill
        performance = np.clip(np.random.normal(true_skill, noise_std), 0, 1)
        
        # Add performance to community sliding window
        self.community_performance_window.append(performance)
        
        # Add to individual sliding window
        rating_data = {
            'timestamp': self.current_time,
            'performance': performance,
            'true_skill': true_skill
        }
        self.rating_windows[agent_id].append(rating_data)
        
        # Calculate current community base rate and individual reputations
        community_base_rate = self.get_current_community_base_rate()
        reputation_sliding = self.get_individual_reputation_sliding_window(agent_id)
        reputation_aging = self.get_individual_reputation_exponential_aging(agent_id)
        
        # Store complete history
        self.complete_history[agent_id].append({
            'time': self.current_time,
            'performance': performance,
            'true_skill': true_skill,
            'reputation_sliding_window': reputation_sliding,
            'reputation_exponential_aging': reputation_aging,
            'community_base_rate': community_base_rate,
            'reputation_vs_community_sliding': reputation_sliding - community_base_rate,
            'reputation_vs_community_aging': reputation_aging - community_base_rate,
            'n_ratings_in_window': len(self.rating_windows[agent_id]),
            'community_window_size': len(self.community_performance_window)
        })
        
        # Update community history
        if agent_id == 0:  # Only update once per time step
            self.community_history.append({
                'time': self.current_time,
                'community_base_rate': community_base_rate,
                'community_window_size': len(self.community_performance_window)
            })

def simulate_dynamic_community_marketplace(n_agents=4, n_time_steps=100, window_size=15, 
                                         community_window_size=25, longevity_factor=0.85, seed=42):
    """
    Simulate marketplace with dynamic community base rate on sliding window.
    """
    np.random.seed(seed)
    
    rep_system = DynamicCommunityReputationSystem(
        n_agents=n_agents, 
        window_size=window_size,
        community_window_size=community_window_size,
        longevity_factor=longevity_factor
    )
    
    print(f"🚀 Dynamic Community Simulation: {n_agents} agents, {n_time_steps} steps")
    print(f"📊 Individual window: {window_size}, Community window: {community_window_size}")
    print(f"🔄 Longevity factor: {longevity_factor}")
    
    # Get initial skill levels for collection period
    initial_skills = [rep_system.sigmoid_skill_evolution(agent_id, 0) 
                     for agent_id in range(n_agents)]
    
    # Simulation loop
    for t in range(n_time_steps):
        rep_system.current_time = t
        
        for agent_id in range(n_agents):
            if t < window_size:
                # Initial collection period - constant skill
                true_skill = initial_skills[agent_id]
            else:
                # Evolution period - skill development
                adjusted_time = t - window_size
                true_skill = rep_system.sigmoid_skill_evolution(agent_id, adjusted_time)
            
            rep_system.update_reputation(agent_id, true_skill)
        
        # Progress reporting
        if t % 20 == 0:
            phase = "Collection" if t < window_size else "Evolution"
            community_rate = rep_system.get_current_community_base_rate()
            avg_skill = np.mean([initial_skills[i] if t < window_size 
                               else rep_system.sigmoid_skill_evolution(i, t - window_size) 
                               for i in range(n_agents)])
            print(f"⏱️  Step {t:3d} ({phase:10s}): Avg skill={avg_skill:.3f}, "
                  f"Community rate={community_rate:.3f}")
    
    return rep_system

def create_comprehensive_visualizations(rep_system):
    """Create comprehensive visualizations with performance vs skill vs reputation analysis."""
    
    fig = plt.figure(figsize=(20, 16))
    colors = plt.cm.Set1(np.linspace(0, 1, rep_system.n_agents))
    
    # 1. Skill Evolution Over Time
    ax1 = plt.subplot(3, 4, 1)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            skills = [h['true_skill'] for h in history]
            ax1.plot(times, skills, '-', color=colors[agent_id], 
                    label=f'Agent {agent_id}', linewidth=2, alpha=0.8)
    
    ax1.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5, label='Evolution Start')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('True Skill Level')
    ax1.set_title('Agent Skill Evolution (Sigmoid)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dynamic Community Base Rate Evolution
    ax2 = plt.subplot(3, 4, 2)
    if rep_system.community_history:
        times = [h['time'] for h in rep_system.community_history]
        community_rates = [h['community_base_rate'] for h in rep_system.community_history]
        ax2.plot(times, community_rates, 'k-', linewidth=3, alpha=0.7, 
                label='Community Base Rate')
        
        # Show individual skill averages for comparison
        avg_skills = []
        for t in times:
            if t < rep_system.window_size:
                avg_skill = np.mean([rep_system.sigmoid_skill_evolution(i, 0) 
                                   for i in range(rep_system.n_agents)])
            else:
                avg_skill = np.mean([rep_system.sigmoid_skill_evolution(i, t - rep_system.window_size) 
                                   for i in range(rep_system.n_agents)])
            avg_skills.append(avg_skill)
        
        ax2.plot(times, avg_skills, 'r--', linewidth=2, alpha=0.7, 
                label='Avg True Skill')
    
    ax2.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Community Base Rate')
    ax2.set_title(f'Dynamic Community Base Rate (Window={rep_system.community_window_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance vs Skill vs Reputation (Sliding Window)
    ax3 = plt.subplot(3, 4, 3)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            performances = [h['performance'] for h in history]
            skills = [h['true_skill'] for h in history]
            reps = [h['reputation_sliding_window'] for h in history]
            
            # Performance (scatter)
            ax3.scatter(times, performances, color=colors[agent_id], alpha=0.3, s=8)
            # Skill (solid line)
            ax3.plot(times, skills, '--', color=colors[agent_id], linewidth=2, alpha=0.7)
            # Reputation (thick line)
            ax3.plot(times, reps, '-', color=colors[agent_id], linewidth=3, 
                    label=f'Agent {agent_id}' if agent_id == 0 else '')
    
    ax3.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Performance/Skill/Reputation')
    ax3.set_title('Performance•Skill--Reputation— (Sliding)')
    if rep_system.n_agents <= 4:
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance vs Skill vs Reputation (Exponential Aging)
    ax4 = plt.subplot(3, 4, 4)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            performances = [h['performance'] for h in history]
            skills = [h['true_skill'] for h in history]
            reps = [h['reputation_exponential_aging'] for h in history]
            
            ax4.scatter(times, performances, color=colors[agent_id], alpha=0.3, s=8)
            ax4.plot(times, skills, '--', color=colors[agent_id], linewidth=2, alpha=0.7)
            ax4.plot(times, reps, '-', color=colors[agent_id], linewidth=3)
    
    ax4.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Performance/Skill/Reputation')
    ax4.set_title(f'Performance•Skill--Reputation— (Aging λ={rep_system.longevity_factor})')
    ax4.grid(True, alpha=0.3)
    
    # 5. Reputation Lag Analysis
    ax5 = plt.subplot(3, 4, 5)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            lag_sliding = [abs(h['true_skill'] - h['reputation_sliding_window']) for h in history]
            lag_aging = [abs(h['true_skill'] - h['reputation_exponential_aging']) for h in history]
            
            ax5.plot(times, lag_sliding, '-', color=colors[agent_id], alpha=0.7,
                    label=f'Sliding Agent {agent_id}' if agent_id < 2 else '')
            ax5.plot(times, lag_aging, '--', color=colors[agent_id], alpha=0.7,
                    label=f'Aging Agent {agent_id}' if agent_id < 2 else '')
    
    ax5.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('|True Skill - Reputation|')
    ax5.set_title('Reputation Lag Behind True Skill')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Individual vs Community Performance (Sliding)
    ax6 = plt.subplot(3, 4, 6)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            rel_reps = [h['reputation_vs_community_sliding'] for h in history]
            community_rates = [h['community_base_rate'] for h in history]
            
            ax6.plot(times, rel_reps, '-', color=colors[agent_id], 
                    linewidth=2, alpha=0.8, label=f'Agent {agent_id}')
    
    ax6.axhline(0, color='black', linestyle='-', alpha=0.3, label='Community Average')
    ax6.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Time Steps')
    ax6.set_ylabel('Reputation - Community Rate')
    ax6.set_title('Individual vs Community (Sliding)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Individual vs Community Performance (Aging)
    ax7 = plt.subplot(3, 4, 7)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            times = [h['time'] for h in history]
            rel_reps = [h['reputation_vs_community_aging'] for h in history]
            
            ax7.plot(times, rel_reps, '-', color=colors[agent_id], 
                    linewidth=2, alpha=0.8, label=f'Agent {agent_id}')
    
    ax7.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax7.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax7.set_xlabel('Time Steps')
    ax7.set_ylabel('Reputation - Community Rate')
    ax7.set_title('Individual vs Community (Aging)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Correlation Analysis: Reputation vs True Skill
    ax8 = plt.subplot(3, 4, 8)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if len(history) > rep_system.window_size:  # Only after collection period
            evolution_history = history[rep_system.window_size:]
            skills = [h['true_skill'] for h in evolution_history]
            reps_sliding = [h['reputation_sliding_window'] for h in evolution_history]
            
            ax8.scatter(skills, reps_sliding, color=colors[agent_id], alpha=0.6, s=15,
                       label=f'Agent {agent_id}')
    
    # Perfect correlation line
    ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Correlation')
    ax8.set_xlabel('True Skill Level')
    ax8.set_ylabel('Reputation (Sliding Window)')
    ax8.set_title('Reputation-Skill Correlation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Performance Variability Analysis
    ax9 = plt.subplot(3, 4, 9)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if len(history) > 10:
            performances = [h['performance'] for h in history]
            skills = [h['true_skill'] for h in history]
            
            # Calculate rolling standard deviation of performance
            window_perf = 10
            rolling_std = []
            times_std = []
            for i in range(window_perf, len(performances)):
                std_val = np.std(performances[i-window_perf:i])
                rolling_std.append(std_val)
                times_std.append(history[i]['time'])
            
            ax9.plot(times_std, rolling_std, '-', color=colors[agent_id], 
                    linewidth=2, alpha=0.8, label=f'Agent {agent_id}')
    
    ax9.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax9.set_xlabel('Time Steps')
    ax9.set_ylabel(f'Performance Std Dev (rolling {10})')
    ax9.set_title('Performance Variability Over Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Method Comparison Heatmap
    ax10 = plt.subplot(3, 4, 10)
    comparison_data = []
    agent_names = []
    
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            final_skill = history[-1]['true_skill']
            final_rep_sliding = history[-1]['reputation_sliding_window']
            final_rep_aging = history[-1]['reputation_exponential_aging']
            final_community_rate = history[-1]['community_base_rate']
            
            comparison_data.append([
                final_skill,
                final_rep_sliding, 
                final_rep_aging,
                final_community_rate,
                final_rep_sliding - final_community_rate,
                final_rep_aging - final_community_rate
            ])
            agent_names.append(f'Agent {agent_id}')
    
    if comparison_data:
        comparison_matrix = np.array(comparison_data).T
        sns.heatmap(comparison_matrix, 
                   xticklabels=agent_names,
                   yticklabels=['True Skill', 'Rep Sliding', 'Rep Aging', 
                              'Community Rate', 'Sliding Δ', 'Aging Δ'],
                   annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax10, cbar_kws={'shrink': .8})
    ax10.set_title('Final State Comparison')
    
    # 11. Skill Growth Rate vs Reputation Adaptation
    ax11 = plt.subplot(3, 4, 11)
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if len(history) > rep_system.window_size + 5:
            evolution_history = history[rep_system.window_size:]
            skills = [h['true_skill'] for h in evolution_history]
            reps = [h['reputation_sliding_window'] for h in evolution_history]
            
            # Calculate growth rates
            skill_growth_rates = np.diff(skills)
            rep_growth_rates = np.diff(reps)
            
            ax11.scatter(skill_growth_rates, rep_growth_rates, color=colors[agent_id], 
                        alpha=0.6, s=20, label=f'Agent {agent_id}')
    
    # Perfect adaptation line
    max_val = 0.02
    ax11.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.5, linewidth=1)
    ax11.set_xlabel('Skill Growth Rate')
    ax11.set_ylabel('Reputation Growth Rate')
    ax11.set_title('Adaptation Rate Correlation')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Community Window Size Effect Analysis
    ax12 = plt.subplot(3, 4, 12)
    if rep_system.community_history:
        times = [h['time'] for h in rep_system.community_history]
        window_sizes = [h['community_window_size'] for h in rep_system.community_history]
        community_rates = [h['community_base_rate'] for h in rep_system.community_history]
        
        # Plot community base rate vs window fill
        colors_window = [plt.cm.viridis(w / rep_system.community_window_size) for w in window_sizes]
        scatter = ax12.scatter(times, community_rates, c=window_sizes, 
                              cmap='viridis', s=30, alpha=0.7)
        
        plt.colorbar(scatter, ax=ax12, label='Community Window Fill')
    
    ax12.axvline(rep_system.window_size, color='gray', linestyle=':', alpha=0.5)
    ax12.set_xlabel('Time Steps')
    ax12.set_ylabel('Community Base Rate')
    ax12.set_title('Community Rate vs Window Fill')
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive analysis
    print("\n" + "="*90)
    print("🏘️  DYNAMIC COMMUNITY REPUTATION SYSTEM ANALYSIS")
    print("="*90)
    
    print(f"\n📊 SYSTEM PARAMETERS:")
    print(f"   • Individual Window Size: {rep_system.window_size}")
    print(f"   • Community Window Size: {rep_system.community_window_size}")
    print(f"   • Longevity Factor (λ): {rep_system.longevity_factor}")
    print(f"   • Final Community Base Rate: {rep_system.get_current_community_base_rate():.4f}")
    print(f"   • Number of Agents: {rep_system.n_agents}")
    print(f"   • Total Time Steps: {len(rep_system.complete_history[0])} steps")
    
    print(f"\n🎯 FINAL PERFORMANCE vs DYNAMIC COMMUNITY:")
    print(f"{'Agent':<8} {'True Skill':<12} {'Sliding Rep':<12} {'Aging Rep':<12} {'Community':<12} {'Sliding Δ':<12} {'Aging Δ':<12}")
    print("-" * 95)
    
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            final_entry = history[-1]
            final_skill = final_entry['true_skill']
            final_rep_sliding = final_entry['reputation_sliding_window']
            final_rep_aging = final_entry['reputation_exponential_aging']
            final_community = final_entry['community_base_rate']
            sliding_delta = final_entry['reputation_vs_community_sliding']
            aging_delta = final_entry['reputation_vs_community_aging']
            
            print(f"Agent {agent_id:<3} {final_skill:<12.4f} {final_rep_sliding:<12.4f} "
                  f"{final_rep_aging:<12.4f} {final_community:<12.4f} {sliding_delta:<+12.4f} {aging_delta:<+12.4f}")
    
    print(f"\n⏱️  ADAPTATION ANALYSIS (Evolution Period Only):")
    print(f"{'Agent':<8} {'Skill Growth':<12} {'Sliding Lag':<12} {'Aging Lag':<12} {'Perf Std':<12} {'Better Method':<15}")
    print("-" * 85)
    
    for agent_id in range(rep_system.n_agents):
        history = rep_system.complete_history[agent_id]
        if history:
            initial_skill = history[0]['true_skill']
            final_skill = history[-1]['true_skill']
            skill_growth = final_skill - initial_skill
            
            # Analysis for evolution period only
            evolution_history = [h for h in history if h['time'] >= rep_system.window_size]
            if evolution_history:
                avg_lag_sliding = np.mean([abs(h['true_skill'] - h['reputation_sliding_window']) 
                                         for h in evolution_history])
                avg_lag_aging = np.mean([abs(h['true_skill'] - h['reputation_exponential_aging']) 
                                       for h in evolution_history])
                
                performances = [h['performance'] for h in evolution_history]
                perf_std = np.std(performances)
                
                better_method = "Sliding" if avg_lag_sliding < avg_lag_aging else "Aging"
                
                print(f"Agent {agent_id:<3} {skill_growth:<+12.4f} {avg_lag_sliding:<12.4f} "
                      f"{avg_lag_aging:<12.4f} {perf_std:<12.4f} {better_method:<15}")
    
    print(f"\n🔍 DYNAMIC COMMUNITY INSIGHTS:")
    print(f"   • Community Base Rate: Updates continuously on sliding window")
    print(f"   • Individual Reputation: Uses dynamic community rate as Bayesian prior")
    print(f"   • Relative Performance: Normalized against evolving community baseline")
    print(f"   • Temporal Adaptation: Both methods adapt to changing community standards")
    print(f"   • Window Effects: Community window size affects adaptation speed")
    
    # Performance correlation analysis
    if rep_system.community_history:
        initial_community_rate = rep_system.community_history[rep_system.window_size]['community_base_rate'] if len(rep_system.community_history) > rep_system.window_size else rep_system.community_history[0]['community_base_rate']
        final_community_rate = rep_system.community_history[-1]['community_base_rate']
        community_growth = final_community_rate - initial_community_rate
        
        print(f"\n📈 COMMUNITY EVOLUTION:")
        print(f"   • Initial Community Rate: {initial_community_rate:.4f}")
        print(f"   • Final Community Rate: {final_community_rate:.4f}")
        print(f"   • Community Growth: {community_growth:+.4f}")
        print(f"   • Community reflects overall skill improvement in marketplace")

def main():
    """Run dynamic community-based Bayesian reputation system simulation."""
    print("🏘️  DYNAMIC COMMUNITY-BASED BAYESIAN REPUTATION SYSTEM")
    print("=" * 70)
    
    # Run simulation with dynamic community base rate
    rep_system = simulate_dynamic_community_marketplace(
        n_agents=4, 

        n_time_steps=120, 
        window_size=15, 
        community_window_size=25,
        longevity_factor=0.85
    )
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(rep_system)
    
    print(f"\n✅ DYNAMIC COMMUNITY SIMULATION COMPLETE!")
    print(f"📈 Key Features Successfully Implemented:")
    print(f"   ✓ Dynamic community base rate on sliding window")
    print(f"   ✓ Individual reputation vs community normalization")
    print(f"   ✓ Performance vs skill vs reputation tracking")
    print(f"   ✓ Sliding window + exponential aging methods")
    print(f"   ✓ Comprehensive lag and adaptation analysis")
    print(f"   ✓ Community evolution tracking")
    print(f"   ✓ No initial reputation dip (collection period)")

if __name__ == "__main__":
    main()

# %%