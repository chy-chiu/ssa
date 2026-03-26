# %%
from ssa.common import RoundData, ExperimentLog


def format_trace(trace):
    for t in trace:
        print(t[2].reasoning)


def format_trace_history(trace, history):
    for t, h in zip(trace, history):
        print("reasoning: ", t[2].reasoning)
        print("action: ", h)
        print("=====")


import numpy as np
from scipy.stats import rankdata
import pandas as pd


# %%

fp = "logs/llm_baseline_full_p_0.log"
exp_log = ExperimentLog.load(fp)

from collections import defaultdict


# %%
def recovery_score(reward_traces):
    n_agents, n_steps = reward_traces.shape

    # Calculate ranks (lower rank = better performance)
    ranks = np.array([rankdata(-reward_traces[:, t]) for t in range(n_steps)]).T

    # Count upward movements (rank decreases)
    improvements = np.diff(ranks, axis=1) < 0
    recovery_rates = improvements.sum(axis=1) / (n_steps - 1)

    return recovery_rates, ranks.max(axis=1) - ranks.min(axis=1)


def calculate_specialization(skill_vector, init=40):
    skills = np.array(skill_vector, dtype=float) - init
    probabilities = skills / np.sum(skills)
    H = entropy(probabilities)
    H_max = np.log(len(skills))
    return 1 - H / H_max


def get_summary_df(exp_log, fp=""):

    rows = []
    for step, hx in enumerate(exp_log.history):

        for agent_idx, a in enumerate(hx.agent_actions):
            agent_id = exp_log.agent_ids[agent_idx]

            action = a.action

            winning_priority = []

            if action == "train":
                train_target = a.targets[0][0]
                rows.append(dict(step=step, agent_id=agent_id, action=action, train_target=train_target))

            elif action == "bid":

                all_targets = [t[0] for t in a.targets]

                for priority, job_id in enumerate(all_targets[:5]):
                    if hx.matched_jobs.get(job_id, -1) == agent_idx:
                        winning_priority.append(priority + 1)
                if len(all_targets) == 0:
                    rows.append(
                        dict(
                            step=step,
                            agent_id=agent_id,
                            action=action,
                        )
                    )
                    continue

                avg_base_price = np.mean([hx.base_prices[t] for t in all_targets if t in hx.base_prices])
                top_base_price = hx.base_prices.get(all_targets[0], 0)
                winrate = len(winning_priority) / min(len(all_targets), 3)

                rows.append(
                    dict(
                        step=step,
                        agent_id=agent_id,
                        action=action,
                        winrate=winrate,
                        winning_priority=np.mean(winning_priority) if winning_priority else 0,
                        avg_base_price=avg_base_price,
                        top_base_price=top_base_price,
                    )
                )
            elif action == "error":
                rows.append(dict(agent_id=agent_id, action=action, step=step))

    df = pd.DataFrame(rows)
    df = df.join(pd.get_dummies(df["action"]))

    df.groupby("agent_id").train_target.nunique()
    df.query('agent_id=="goog"')
    df.groupby("agent_id").action.value_counts()
    rewards_df = pd.DataFrame(
        [
            dict(run=fp, agent_id=agent.id, reward=agent.total_reward, atype=agent.id.split("-")[0])
            for agent in exp_log.agents
        ]
    ).set_index("agent_id")
    rewards_df["rank"] = rewards_df["reward"].rank(ascending=False, method="average")
    rewards_df["reward_normalized"] = rewards_df["reward"] / rewards_df["reward"].sum()

    summary_df = df.groupby("agent_id").agg(
        {
            "winrate": "mean",
            "top_base_price": "mean",
            "avg_base_price": "mean",
            "bid": "mean",
            "train": "mean",
            "error": "mean",
            "train_target": "nunique",
        }
    )

    all_winning_bids = []
    all_agent_bids = []
    for hx in exp_log.history:
        agent_bids = defaultdict(list)
        agent_winning_bids = defaultdict(list)

        for job_id, bids in hx.agent_bids_normalized.items():
            for agent_idx, nbid in bids.items():
                agent_id = exp_log.agent_ids[agent_idx]
                agent_bids[agent_id].append(nbid)
                if hx.matched_jobs.get(job_id, -1) == agent_idx:
                    agent_winning_bids[agent_id].append(nbid)

        all_winning_bids.append({k: np.mean(b) for k, b in agent_winning_bids.items()})
        all_agent_bids.append({k: np.mean(b) for k, b in agent_bids.items()})

    winning_bids = pd.DataFrame(all_winning_bids).mean()
    agent_bids = pd.DataFrame(all_agent_bids).mean()

    summary_df["all_bids"] = agent_bids
    summary_df["winning_bids"] = winning_bids

    winning_priority = df.query("winrate > 0").groupby("agent_id").winning_priority.mean()

    summary_df["win_prio"] = df.query("winrate > 0").groupby("agent_id").winning_priority.mean()
    completion_token = pd.Series(
        {
            agent_id: t["total_token_usage"]["completion_tokens"]
            for agent_id, t in zip(exp_log.agent_ids, exp_log.token_usage["agent_token_usage"])
        }
    )

    summary_df["completion_tokens"] = completion_token
    total_token = pd.Series(
        {
            agent_id: t["total_token_usage"]["total_tokens"]
            for agent_id, t in zip(exp_log.agent_ids, exp_log.token_usage["agent_token_usage"])
        }
    )

    summary_df["total_tokens"] = total_token

    reward_traces = np.array(exp_log.agent_total_rewards).T
    scores, rank_jump = recovery_score(reward_traces)

    summary_df["recovery"] = pd.Series({exp_log.agent_ids[agent_idx]: a for agent_idx, a in enumerate(scores)})
    summary_df["rank_jump"] = pd.Series({exp_log.agent_ids[agent_idx]: a for agent_idx, a in enumerate(rank_jump)})
    summary_df["train_p"] = summary_df["train"] / (summary_df["train"] + summary_df["bid"])

    skill_rep_dicts = []

    for agent in exp_log.agents:
        agent_skills = [sk[1][-1] for sk in agent.skill_history.values()]
        agent_reputation = [r[1] for r in agent.reputation.values()]

        skill_rep_dicts.append(
            dict(
                agent_id=agent.id,
                skill_sum=sum(agent_skills),
                skill_max=max(agent_skills),
                skill_spec=calculate_specialization(agent_skills),
                rep_avg=np.mean(agent_reputation),
                rep_max=max(agent_reputation),
                rep_spec=calculate_specialization(agent_reputation, init=0),
            )
        )

    skill_rep_df = pd.DataFrame(skill_rep_dicts).set_index("agent_id")

    return (
        rewards_df.join(summary_df)
        .join(skill_rep_df)
        .reset_index()[
            [
                "run",
                "agent_id",
                "atype",
                "reward",
                "reward_normalized",
                "rank",
                "winrate",
                "win_prio",
                "recovery",
                "rank_jump",
                "top_base_price",
                "avg_base_price",
                "all_bids",
                "winning_bids",
                "train_p",
                "train_target",
                "skill_sum",
                "skill_max",
                "skill_spec",
                "rep_avg",
                "rep_max",
                "rep_spec",
                "total_tokens",
                "completion_tokens",
            ]
        ]
    )


# %%
from scipy.stats import entropy

# %%
# %%

# %%
# %%


# %%
agent.reputation
# %%
df = get_summary_df(exp_log, fp)
df

# %%

# %%
"""Cumulative Reward 
(and cumulative normalized reward)
Period Reward
Rank - Average and Final - ? done
Specialization
Consistency (e.g. same or different actions??)
Train %
Total skill level
Win rate
Risk aversion (as measured by rank of price jobs..), top and average
win priority - are you winning the good jobs or low priority jobs? 
Ability to recover (rank changes)
"""


# %%
# Calculate ranks (lower rank = better performance)
n_steps = 100
ranks = np.array([rankdata(-reward_traces[:, t]) for t in range(n_steps)]).T

# Count upward movements (rank decreases)
improvements = np.diff(ranks, axis=1) < 0
# %%
improvements
# %%
import matplotlib.pyplot as plt

plt.plot()
# %%


# %%

## Calculate: win rate, risk aversity, consistency
exp_log.agent_ids

hx = exp_log.history[0]
# %%
# %%
