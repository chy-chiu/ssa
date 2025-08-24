
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from typing import List, Dict


def plot_allocation(ax, allocation: List[Dict[str, str]]):
    """
    Plot task allocation over time as horizontal bars.

    Parameters:
    ax: matplotlib axis object
    allocation: list of dict[str, str] where each dict represents timestep allocations

    Returns:
    ax: the modified axis object
    """

    # Extract all unique tasks and agents
    all_tasks = set()
    all_agents = set()

    for timestep in allocation:
        all_tasks.update(timestep.keys())
        all_agents.update(timestep.values())

    all_tasks = sorted(list(all_tasks), reverse=True)
    all_agents = sorted(list(all_agents))

    # Create color palette for agents
    colors = sns.color_palette("tab10", n_colors=len(all_agents))
    agent_colors = dict(zip(all_agents, colors))

    # Create y-axis positions for tasks
    y_positions = {task: i for i, task in enumerate(all_tasks)}

    # Plot allocation for each task
    bar_height = 0.8  # Height of each task bar

    for task in all_tasks:
        y_pos = y_positions[task]

        # Group consecutive timesteps with same agent
        current_agent = None
        start_time = None

        for timestep, assignments in enumerate(allocation):
            agent = assignments.get(task, None)

            if agent != current_agent:
                # End previous segment if exists
                if current_agent is not None and start_time is not None:
                    width = timestep - start_time
                    ax.barh(
                        y_pos,
                        width,
                        left=start_time,
                        height=bar_height,
                        color=agent_colors[current_agent],
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=0.5,
                    )

                # Start new segment
                current_agent = agent
                start_time = timestep

        # Handle the last segment
        if current_agent is not None and start_time is not None:
            width = len(allocation) - start_time
            ax.barh(
                y_pos,
                width,
                left=start_time,
                height=bar_height,
                color=agent_colors[current_agent],
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )

    # Customize the plot
    ax.set_yticks(range(len(all_tasks)))
    ax.set_yticklabels(all_tasks)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Tasks")
    ax.set_xlim(0, len(allocation))
    ax.grid(axis="x", alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=agent_colors[agent], alpha=0.8, label=agent) for agent in all_agents
    ]
    ax.legend(handles=legend_elements)

    # Set title
    ax.set_title("Task Allocation Over Time")

    return ax


def plot_agent_trace(ax, agent_traces: np.ndarray, label: List[str]):
    """Takes in any agent trace over multiple runs in shape (n_agents, n_runs, n_steps) and plot the mean / std"""
    
    for agent_idx, agent_trace in enumerate(agent_traces):
        
        ax.plot(agent_trace.mean(axis=0).T, label=label[agent_idx])

        ax.fill_between(
            np.arange(agent_trace.shape[-1]),
            agent_trace.mean(axis=0).T - agent_trace.std(axis=0).T,
            agent_trace.mean(axis=0).T + agent_trace.std(axis=0).T,
            alpha=0.1,
        )

    ax.legend()
    return ax
