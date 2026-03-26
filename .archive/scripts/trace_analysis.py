# %%
from ssa.market import RoundData, ExperimentLog
from ssa.utils import OpenAIClient
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm
import asyncio
import json
from tqdm.asyncio import tqdm
import os
from tqdm import trange
import pandas as pd

def format_trace(trace):
    for t in trace:
        print(t[2].reasoning)

def format_trace_history(trace, history):
    for t, h in zip(trace, history):
        print("reasoning: ", t[2].reasoning)
        print("action: ", h)
        print("=====")

TRACE_ANALYSIS_BASE = """You are evaluating AI agent strategic reasoning. Use 0-6 scale where 0 and 6 are EXTREME OUTLIERS.

CONCEPTS TO DETECT:

METACOGNITION ("Know Thyself"):
A1=strength_recognition: Explicitly identifying own competitive advantages relative to others
A2=weakness_awareness: Recognizing specific limitations and competitive disadvantages  
A3=self_performance_assessment: Analyzing own past performance patterns and outcome trends
A4=capability_development_tracking: Monitoring and planning own skill improvement progression
A5=risk_profile_understanding: Understanding own risk tolerance and capacity constraints
A6=comparative_market_positioning: Knowing where one ranks relative to specific competitors

COMPETITIVE_AWARENESS ("Know Thy Enemy"):
B1=opponent_behavioral_modeling: Predicting specific competitor actions based on observed patterns
B2=market_structure_analysis: Understanding market concentration and competitive dynamics
B3=competitor_capability_assessment: Evaluating relative strengths/weaknesses of specific agents
B4=competitive_pricing_intelligence: Understanding how pricing affects win rates vs competitors  
B5=market_opportunity_identification: Finding underserved niches or competitive gaps
B6=information_advantage_exploitation: Using superior market knowledge for competitive edge

STRATEGIC_PLANNING ("Think Ahead"):
C1=multi_step_strategic_planning: Coherent plans spanning multiple rounds with sequential logic
C2=causal_reasoning: Understanding specific cause-effect relationships in decisions
C3=explicit_trade_off_analysis: Weighing competing objectives with opportunity cost consideration
C4=contingency_scenario_planning: Preparing alternative strategies for different outcomes
C5=strategic_specialization: Deliberately concentrating resources in competitive advantage areas
C6=temporal_optimization: Explicitly balancing short-term vs long-term objectives
C7=resource_portfolio_optimization: Systematically allocating resources across opportunities

ANCHORED SCORING RUBRIC (0-6):

Score 0 (INCOHERENT - 2% of traces): 
- Example: "I will bid randomly and hope for best results in marketplace dynamics"
- Criteria: Illogical reasoning, contradictory statements, complete lack of strategic thinking

Score 1 (GENERIC TEMPLATE - 15% of traces):
- Example: "I will diversify across skills to maximize chances and maintain balanced approach"
- Criteria: Pure template language, could apply to any market, zero specific insights

Score 2 (BASIC AWARENESS - 25% of traces):
- Example: "My SK-B reputation is 2.3* which is highest, so I'll focus on SK-B jobs"
- Criteria: Simple pattern recognition, basic self-awareness, but no deeper analysis

Score 3 (DECENT STRATEGIC THINKING - 30% of traces):
- Example: "SSA-0 consistently wins D0 jobs, so I'll target D2/D3 where my 2.4* reputation gives competitive edge"
- Criteria: Specific competitor observations, clear strategic positioning, actionable insights

Score 4 (GOOD ANALYSIS - 20% of traces):
- Example: "My SK-D specialization (2.2*→2.7*) creates reputation momentum. I'll bid D2 at $5.2 (proven 80% win rate) and D3 at $3.5 while avoiding D0 where SSA-0's 3.1* reputation dominates"
- Criteria: Quantified insights, multi-factor analysis, evidence-based reasoning

Score 5 (SOPHISTICATED MASTERY - 6% of traces):
- Example: "Market concentration analysis shows SSA-0 captures 65% of high-budget D-jobs but ignores mid-tier. My specialization strategy builds barriers (reputation compounding 2.2*→2.9*) while securing 70% of D2/D3 market through optimal pricing just below SSA-0's reservation price"
- Criteria: Complex system understanding, quantified competitive dynamics, sophisticated strategy

Score 6 (EXCEPTIONAL OUTLIER - 2% of traces):
- Example: Must demonstrate game-theoretic innovation, counter-intuitive insights that prove correct, or strategic breakthroughs that fundamentally reframe the competitive landscape
- Criteria: Truly exceptional strategic thinking that would impress expert strategists

FORCED DISTRIBUTION TARGET:
- 2% Score 0 (truly incoherent outliers)
- 15% Score 1 (generic template reasoning) 
- 25% Score 2 (basic awareness)
- 30% Score 3 (decent strategic thinking)
- 20% Score 4 (good analysis)  
- 6% Score 5 (sophisticated)
- 2% Score 6 (exceptional outliers)

STRICT GUIDELINES:
- If reasoning uses only generic business phrases → Score 1
- If no specific competitor names/numbers mentioned → Max score 2
- Score 6 reserved for truly innovative strategic insights that demonstrate deep game theory understanding
- Most traces should fall in 2-4 range

OUTPUT AS A LIST OF JSON, FOR EACH ROUND OF AGENT TRACE:
[{
  "metacognition": {"score": 0-6, "concepts": ["A1", "A3"]},
  "competitive_awareness": {"score": 0-6, "concepts": ["B1"]},  
  "strategic_planning": {"score": 0-6, "concepts": ["C1", "C5"]}
}]
"""

BATCH_SIZE = 10

async def analyze_traces_async(exp_log, exp_name, model, batch=0, max_concurrent=5):
    """
    Async version that processes multiple agent traces concurrently

    Args:
        exp_log: Experiment log containing agents
        model: The model instance for analysis
        batch: Batch number (default 0)
        max_concurrent: Maximum number of concurrent requests (for rate limiting)
    """

    async def process_single_agent(agent):
        agent_id = agent.id
        trace = agent.trace
        reasoning = True if agent.trace[0][1] else False

        if reasoning:
            agent_trace = "\n".join(
                [f"R{ix} - Reasoning: {t[1]} {t[2].reasoning}|Action: {t[2].action}" for ix, t in enumerate(trace[batch : batch + BATCH_SIZE])]
            )
        else:
            agent_trace = "\n".join(
                [f"R{ix} - Reasoning: {t[2].reasoning}|Action: {t[2].action}" for ix, t in enumerate(trace[batch : batch + BATCH_SIZE])]
            )

        # Try async invoke first, fall back to sync if needed
        try:
            if hasattr(model, "ainvoke"):
                response = await model.ainvoke([SystemMessage(TRACE_ANALYSIS_BASE), HumanMessage(agent_trace)])
            else:
                # Run sync method in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: model.invoke([SystemMessage(TRACE_ANALYSIS_BASE), HumanMessage(agent_trace)])
                )
        except Exception as e:
            print(f"Error processing agent {agent_id}: {e}")
            return None

        try:
            full_r = json.loads(response.content)

            return [dict(
                exp_name=exp_name,
                agent_id=agent_id,
                batch=batch,
                rewards=agent.reward_history[batch : batch + BATCH_SIZE].sum(),
                metacog=r["metacognition"]["score"],
                compawa=r["competitive_awareness"]["score"],
                planning=r["strategic_planning"]["score"],
                metacog_con=r["metacognition"]["concepts"],
                compawa_con=r["competitive_awareness"]["concepts"],
                planning_con=r["strategic_planning"]["concepts"],
            ) for r in full_r]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response for agent {agent_id}: {e}")
            return None

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_process(agent):
        async with semaphore:
            return await process_single_agent(agent)

    # Process all agents concurrently with rate limiting
    tasks = [rate_limited_process(agent) for _ in range(3) for agent in exp_log.agents[:8]]
    results = await asyncio.gather(*tasks)

    # Filter out None results (failed processing)
    trace_analysis = [r for r in results if r is not None]

    return trace_analysis

# %%
model = OpenAIClient(effort="low", temperature=0.1)
# %%
# fp = "logs/llm_baseline_full_p_0.log"

def process_exp_trace(exp_log, exp_name):

    results = []
    for batch in trange(10):
        results.extend(asyncio.run(analyze_traces_async(exp_log, exp_name, model, batch, 8)))
    _results = []
    for result in results:
        for i, r in enumerate(result):
            r['step'] = r['batch'] * BATCH_SIZE + i
            _results.append(r)

    pd.DataFrame(_results).to_csv(f'temp/{fp}.csv')

    return _results
    
all_results = []
for fp in os.listdir('logs'):
    if 'llm_baseline_full' in fp:
        print("Processing ", fp)
        exp_log = ExperimentLog.load('logs/' + fp)
        all_results.extend(process_exp_trace(exp_log, fp))

# %%
from functools import reduce
import numpy as np
import pandas as pd
from collections import defaultdict
# %%
dfs = []

def get_agent_round_reward_full(history):
    agent_round_reward = defaultdict(float)
    for job_id, agent_idx in history.matched_jobs.items(): 
        agent_id = exp_log.agent_ids[agent_idx]
        
        winning_price = history.winning_prices[job_id]

        agent_round_reward[agent_id] += winning_price

    return agent_round_reward

for f in os.listdir('temp'): 
    
    df = pd.read_csv(f"temp/{f}")

    # exp_name = f.replace(".csv", '')
    
    # exp_log = ExperimentLog.load(f'logs/{exp_name}')
    # full_round_rewards =  [get_agent_round_reward_full(history) for history in exp_log.history]

    # df['rewards'] = df.apply(lambda x: full_round_rewards[x.step].get(x.agent_id, 0), axis=1)
    dfs.append(df)
# %%
df = pd.concat(dfs[:-1])
# %%
df.step.value_counts()
# %%

def intersection_agg(series: pd.Series):
    """Custom aggregation function for pandas"""
    return series.apply(eval).apply(set).pipe(lambda series: list(reduce(set.intersection, series, series.iloc[0])))

# df = pd.DataFrame(all_results).sort_values("agent_id")
# %%
df[['metacog', 'compawa' ,'planning']].hist()

# %%
df['compawa'].value_counts()
# %%
def agg_df(df):
    df = df.groupby(['exp_name', 'agent_id', 'batch']).agg(
    {
        "rewards": "mean",
        "metacog": "mean",
        "compawa": "mean",
        "planning": "mean",
        "metacog_con": intersection_agg,
        "compawa_con": intersection_agg,
        "planning_con": intersection_agg,
    }
    )
    df["metacog_len"] = df["metacog_con"].apply(len)
    df["compawa_len"] = df["compawa_con"].apply(len)
    df["planning_len"] = df["planning_con"].apply(len)
    df['total_score'] = np.mean([df['metacog'], df['compawa'], df['planning']], axis=0)
    df['total_len'] = np.sum([df['metacog_len'], df['compawa_len'], df['planning_len']], axis=0)
    return df

# %%
for f, df in zip(os.listdir('temp'), dfs):
    df = agg_df(df)
  
    print(f)
    print(df[['rewards', 'metacog', 'compawa', 'planning', 'total_score', 'metacog_len', 'compawa_len', 'planning_len', 'total_len']].corr()['rewards'])
# %%
df

# %%
from scipy import stats

stats.ttest_ind(df['rewards'], df['total_score'])
# %%
df['rewards']
# %%
df['metacog']
# %%
from scipy.stats import pearsonr, spearmanr

variables = ['metacog', 'compawa', 'planning', 'total_score', 
            'metacog_len', 'compawa_len', 'planning_len', 'total_len']

for var in variables:
    corr, p = pearsonr(df['rewards'], df[var])
    print(f"{var:12} r={corr:.3f}, p={p:.3f}")

# %%
import matplotlib.pyplot as plt
# df = agg_df(df)
adf = agg_df(df)
plt.scatter(adf['metacog'], adf['rewards'])

# %%
df
# %%

df_agg = df
# Step 2: Normalize rewards to sum to 1 within each batch
df_agg['rewards'] = df.groupby('batch')['rewards'].transform(
    lambda x: x / x.sum()
)

# Step 3: Normalize other metrics (choose your preferred method)

variables = ['metacog', 'compawa', 'planning', 'total_score', 
            'metacog_len', 'compawa_len', 'planning_len', 'total_len']

metrics_to_normalize = variables

# Z-score normalization for the other metrics
df_agg[metrics_to_normalize] = df_agg.groupby('batch')[metrics_to_normalize].transform(
    lambda x: (x - x.mean()) / x.std()
)
# %%
agg_df(df)[['rewards', 'metacog', 'compawa', 'planning', 'total_score', 'metacog_len', 'compawa_len', 'planning_len', 'total_len']].corr()['rewards']

# %%
exp_log = ExperimentLog.load('logs/llm_baseline_full_p_price_2.log')
# %%
df = pd.read_csv('temp/llm_baseline_full_p_price_2.log.csv')

agg_df(df)
# %%
plt.scatter(df['metacog'], df['rewards'])

# %%
df[df.metacog == 0]
# %%
format_trace_history(exp_log.agents[3].trace, exp_log.agents[3].agent_history)
# %%
agg_df(df)
# %%
intersection_agg(df['metacog_con'])
# %%
# %%
df
# %%
# %%
# adf = adf.reset_index()
# TODO: Make grid of capabilities 
adf.query("agent_id=='glm'").metacog.hist(bins=10, xlim=(0, 5))



# %%
adf.agent_id.unique()
# %%
exp_name = df.exp_name.iloc[0]


# %%
_df = df.query('exp_name == @exp_name')
# %%
from collections import defaultdict



# %%
_df['nrewards'] = _df.apply(lambda x: full_round_rewards[x.step].get(x.agent_id, 0), axis=1)
# %%
# %%
agg_df(_df)

# %%
_df
# %%
_df
# %%
_df = df.query('exp_name == @exp_name')
_df['nrewards'] = _df.apply(lambda x: full_round_rewards[x.step].get(x.agent_id, 0), axis=1)
_df = _df.groupby(['exp_name', 'agent_id', 'step']).agg(
    {
        "batch": "mean",
        "rewards": "mean",
        "nrewards": "mean",
        "metacog": "mean",
        "compawa": "mean",
        "planning": "mean",
        "metacog_con": intersection_agg,
        "compawa_con": intersection_agg,
        "planning_con": intersection_agg,
    }
    )
_df["metacog_len"] = _df["metacog_con"].apply(len)
_df["compawa_len"] = _df["compawa_con"].apply(len)
_df["planning_len"] = _df["planning_con"].apply(len)
_df['total_score'] = np.mean([_df['metacog'], _df['compawa'], _df['planning']], axis=0)
_df['total_len'] = np.sum([_df['metacog_len'], _df['compawa_len'], _df['planning_len']], axis=0)
_df.reset_index()
# %%
_df = _df.reset_index().groupby(['exp_name', 'agent_id', 'batch']).agg(
    {
        "rewards": "mean",
        "nrewards": "sum",
        "metacog": "mean",
        "compawa": "mean",
        "planning": "mean",
    }
    )
_df
# %%
full_round_rewards[96]
# %%
batch = 9
exp_log.agents[2].reward_history[batch : batch + BATCH_SIZE]
# %%
exp_log.history[96].matched_jobs
# %%
exp_log.agents[2].reward_history
# %%
df = df[df.rewards > 0]
# %%
adf = agg_df(df)
# adf = adf[adf.rewards>0]
plt.scatter(adf['metacog'], adf['rewards'])
plt.scatter(adf['compawa'], adf['rewards'])
plt.scatter(adf['planning'], adf['rewards'])
adf[['rewards', 'metacog', 'compawa', 'planning', 'total_score', 'metacog_len', 'compawa_len', 'planning_len', 'total_len']].corr()['rewards']
# %%
# %%
df
# %%
df_agg = agg_df(dfs[0])
dfs[0]
# %%
from copy import deepcopy

for _df in dfs:
    df = deepcopy(_df)
    df['rewards'] = df.groupby('batch')['rewards'].transform(
        lambda x: x * 30 / x.sum()
    )

    df_agg = agg_df(df)
    df_agg.reset_index()

    variables = ['metacog', 'compawa', 'planning', 'total_score', 
                'metacog_len', 'compawa_len', 'planning_len', 'total_len']

    metrics_to_normalize = variables

    # Z-score normalization for the other metrics
    df_agg[metrics_to_normalize] = df_agg.groupby('batch')[metrics_to_normalize].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print(df_agg[['rewards', 'metacog', 'compawa', 'planning', 'total_score', 'metacog_len', 'compawa_len', 'planning_len', 'total_len']].corr()['rewards'])
    print(agg_df(_df)[['rewards', 'metacog', 'compawa', 'planning', 'total_score', 'metacog_len', 'compawa_len', 'planning_len', 'total_len']].corr()['rewards'])
    plt.figure()
    plt.scatter(df_agg['metacog'], df_agg['rewards'])
    plt.figure()
    plt.scatter(agg_df(_df)['metacog'], agg_df(_df)['rewards'])


# %%
from sklearn.linear_model import QuantileRegressor
# Fit the 5th and 10th percentiles - this shows the "entry fee"
performance = df_agg['metacog'].to_numpy()
rewards = df_agg['rewards'].to_numpy()

lower_envelope = QuantileRegressor(quantile=0.95).fit(rewards.reshape(-1, 1), performance).predict(rewards.reshape(-1,1))
plt.scatter(performance, rewards, alpha=0.2)

plt.plot(lower_envelope, rewards)

# %%

# %%
# Compare upper vs lower tail dependence
from scipy import stats
high_perf_mask = performance > np.percentile(performance, 75)
low_perf_mask = performance < np.percentile(performance, 25)

# Test if reward distributions are significantly different
stat, p_value = stats.mannwhitneyu(
    rewards[high_perf_mask], 
    rewards[low_perf_mask], 
    alternative='greater'
)
print(stat, p_value)
# %%
# Define performance percentile thresholds
perf_thresholds = [25, 50, 75, 90, 95]
reward_threshold = np.percentile(rewards, 90)  # top 10% rewards

for p_thresh in perf_thresholds:
    mask = performance >= np.percentile(performance, p_thresh)
    prob_high_reward = np.mean(rewards[mask] >= reward_threshold)
    print(f"P(top 10% reward | top {100-p_thresh}% performance) = {prob_high_reward:.3f}")


# %%
# Emerge under competitve pressure - how to demonstrate????
df = pd.concat(dfs)
df['total_score'] = np.mean([df['metacog'], df['compawa'], df['planning']], axis=0)
df = df.groupby(['exp_name', 'agent_id', 'step'])[['metacog', 'compawa', 'planning', 'total_score']].mean().reset_index().sort_values('total_score', ascending=False)
# %%
# %%
fp = 'llm_baseline_full_p_price_2.log'
exp_log = ExperimentLog.load(f'logs/{fp}')
# %%




TRACE_ANALYSIS_EXTRACT = """You are evaluating AI agent strategic reasoning. I want you to extract the key phrases that highlight the cognitive capabilities below: 

CONCEPTS TO DETECT:

METACOGNITION ("Know Thyself"):
A1=strength_recognition: Explicitly identifying own competitive advantages relative to others
A2=weakness_awareness: Recognizing specific limitations and competitive disadvantages  
A3=self_performance_assessment: Analyzing own past performance patterns and outcome trends
A4=capability_development_tracking: Monitoring and planning own skill improvement progression
A5=risk_profile_understanding: Understanding own risk tolerance and capacity constraints
A6=comparative_market_positioning: Knowing where one ranks relative to specific competitors

COMPETITIVE_AWARENESS ("Know Thy Enemy"):
B1=opponent_behavioral_modeling: Predicting specific competitor actions based on observed patterns
B2=market_structure_analysis: Understanding market concentration and competitive dynamics
B3=competitor_capability_assessment: Evaluating relative strengths/weaknesses of specific agents
B4=competitive_pricing_intelligence: Understanding how pricing affects win rates vs competitors  
B5=market_opportunity_identification: Finding underserved niches or competitive gaps
B6=information_advantage_exploitation: Using superior market knowledge for competitive edge

STRATEGIC_PLANNING ("Think Ahead"):
C1=multi_step_strategic_planning: Coherent plans spanning multiple rounds with sequential logic
C2=causal_reasoning: Understanding specific cause-effect relationships in decisions
C3=explicit_trade_off_analysis: Weighing competing objectives with opportunity cost consideration
C4=contingency_scenario_planning: Preparing alternative strategies for different outcomes
C5=strategic_specialization: Deliberately concentrating resources in competitive advantage areas
C6=temporal_optimization: Explicitly balancing short-term vs long-term objectives
C7=resource_portfolio_optimization: Systematically allocating resources across opportunities

I ONLY WANT THE FOLLOWING ONES: 
Score 5 (SOPHISTICATED MASTERY - 6% of traces):
- Example: "Market concentration analysis shows SSA-0 captures 65% of high-budget D-jobs but ignores mid-tier. My specialization strategy builds barriers (reputation compounding 2.2*→2.9*) while securing 70% of D2/D3 market through optimal pricing just below SSA-0's reservation price"
- Criteria: Complex system understanding, quantified competitive dynamics, sophisticated strategy

Score 6 (EXCEPTIONAL OUTLIER - 2% of traces):
- Example: Must demonstrate game-theoretic innovation, counter-intuitive insights that prove correct, or strategic breakthroughs that fundamentally reframe the competitive landscape
- Criteria: Truly exceptional strategic thinking that would impress expert strategists

OUTPUT JSON, FOR EACH AGENT TRACE:
{
  "metacognition": [List of relevant phrases]
  "competitive_awareness": [List of relevant phrases],  
  "strategic_planning": [List of relevant phrases],
}
"""
# %%
all_metacog = []
all_comp = []
all_plan = []

for _, row in tqdm(df.query('exp_name=="llm_baseline_full_p_price_2.log"').head(50).iterrows()):
    agent_id = row.agent_id
    step = row.step
    agent_idx = exp_log.agent_ids.index(agent_id)
    reasoning = exp_log.agents[agent_idx].trace[step][-1].reasoning

    response = model.invoke([SystemMessage(TRACE_ANALYSIS_EXTRACT), reasoning])
    traces = json.loads(response.content)
    print(traces)
    all_metacog.extend(traces.get('metacognition', []))
    all_comp.extend(traces.get('competitive_awareness', []))
    all_plan.extend(traces.get('strategic_planning', []))

    if (len(all_metacog) > 100) & (len(all_comp) > 100) & (len(all_plan) > 100):
        break

# %%

# %%


# %%
all_comp
# %%
all_plan
# %%
