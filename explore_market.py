# %%
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ssa.agents import ImproveAgent, LLMAgent, OracleAgent, StaticAgent
from ssa.agents.policy import PolicyAgent
from ssa.agents._ssa import LLMSSA
from ssa.galeshapley import multi_galeshapley
from ssa.market import ExperimentLog, Job, LabourMarket
from ssa.tasks import ProxyAgent, ProxyTask
from ssa.tasks.cipher import CipherAgent, CipherTask
from ssa.utils import init_azure_model, init_openrouter_chat_model

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import zscore
import matplotlib.pyplot as plt


# %%
import seaborn as sns

SNS_RED = sns.color_palette("Set1")[0]
SNS_BLUE = sns.color_palette("Set1")[1]
SNS_GREEN = sns.color_palette("Set1")[2]
SNS_CYAN = sns.color_palette("Set2")[0]
SNS_ORANGE = sns.color_palette("Set2")[1]
SNS_BBLUE = sns.color_palette("Set2")[2]
SNS_BPURPLE = sns.color_palette("Set2")[3]
SNS_YELLOW = sns.color_palette("Set2")[5]
SNS_GREY = sns.color_palette("Set2")[7]

# %%
np.random.seed(13123)

from tqdm import tqdm
### Beveridge related stuff
def hyperbolic_func(x, a, b, c):
    """Hyperbolic function: y = a/(x + b) + c"""
    return a / (x + b) + c


def remove_outliers_residual_based_b(X, y, func, params, outlier_percent=5):
    """Remove outliers based on residuals from initial fit"""
    y_pred = func(X, *params)
    residuals = np.abs(y - y_pred)
    threshold = np.percentile(residuals, 100 - outlier_percent)
    mask = residuals <= threshold
    return X[mask], y[mask], mask


def remove_outliers_zscore(X, y, z_threshold=2.5):
    """Remove outliers based on Z-score"""
    z_scores_x = np.abs(zscore(X))
    z_scores_y = np.abs(zscore(y))
    mask = (z_scores_x < z_threshold) & (z_scores_y < z_threshold)
    return X[mask], y[mask], mask


def fit_hyperbola_with_outlier_removal(data, outlier_percent=5, method="residual"):
    """
    Fit hyperbolic curve with outlier removal

    Parameters:
    - data: numpy array of shape (n, 2) or tuple (X, y)
    - outlier_percent: percentage of outliers to remove
    - method: 'residual' or 'zscore'
    """

    # Handle input format
    if isinstance(data, tuple):
        X, y = data
    else:
        X, y = data[:, 0], data[:, 1]

    # Initial fit to identify outliers (if using residual method)
    if method == "residual":
        # Robust initial parameter estimation
        try:
            initial_params = [np.std(y) * np.mean(X), np.mean(X), np.mean(y)]
            popt_initial, _ = curve_fit(hyperbolic_func, X, y, p0=initial_params, maxfev=5000)
        except:
            # Fallback parameters
            popt_initial = [1.0, 1.0, np.mean(y)]

        X_clean, y_clean, mask = remove_outliers_residual_based_b(X, y, hyperbolic_func, popt_initial, outlier_percent)
    else:  # zscore method
        z_threshold = np.sqrt(2 * np.log(100 / outlier_percent))  # Convert percentage to z-score
        X_clean, y_clean, mask = remove_outliers_zscore(X, y, z_threshold)

    # Final fit on cleaned data
    try:
        initial_params = [np.std(y_clean) * np.mean(X_clean), np.mean(X_clean), np.mean(y_clean)]
        popt_final, pcov = curve_fit(hyperbolic_func, X_clean, y_clean, p0=initial_params, maxfev=5000)
    except Exception as e:
        raise RuntimeError(f"Curve fitting failed: {e}")

    # Calculate fit quality metrics
    y_pred = hyperbolic_func(X_clean, *popt_final)
    r_squared = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)
    rmse = np.sqrt(np.mean((y_clean - y_pred) ** 2))

    results = {
        "params": popt_final,
        "covariance": pcov,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "outlier_mask": mask,
        "r_squared": r_squared,
        "rmse": rmse,
        "outliers_removed": len(X) - len(X_clean),
    }

    return results

# GINI RELATED STUFF
# %%
PERIOD = 5
def gini(wealth):
    wealth = np.sort(wealth[wealth >= 0])
    n = len(wealth)
    cumsum = np.cumsum(wealth)
    return 1 - (2 * np.sum(cumsum)) / (n * cumsum[-1]) + 1 / n

total_n_jobs = 64
n_agents = 32
agent_pref_limit = 16
market_pref_limit = 100
# all_r = []
for _ in range(3):
    for market_limit in range(1, 9):
        for job_task_r in [1, 4, 16, 64]:
            n_tasks = total_n_jobs // job_task_r
            n_jobs = total_n_jobs // n_tasks

            task_ids = [f"task_{i}" for i in range(n_tasks)]

            tasks = [ProxyTask(t, noise=0.1) for t in task_ids]

            jobs = [
                Job(
                    id=f"{task_id}_{i}",
                    task_id=task_id,
                    job_p=np.clip(job_p + np.random.normal(0, 0.1), 0, 1),
                    base_reward=10,
                )
                for task_id in task_ids
                for i in range(n_jobs)
            ]

            agents = [PolicyAgent(agent_id=f"pol_{i}", jobs=jobs, model=None, verbose=False) for i in range(n_agents)]

            for agent in agents:
                agent.set_policy(train_p=np.clip(train_p + np.random.normal(0, 0.1), 0, 1))

            market = LabourMarket(
                jobs=jobs,
                market_pref_limit=market_pref_limit,
                agent_pref_limit=agent_pref_limit,
                market_limit=market_limit,
                tasks=tasks,
                agents=agents,
                skill_phi=0.1,
                rep_window=50,
                rep_lambda=0.5,
                rep_sensitivity=2,
                gumbel_t=0.001,
            )

            for _ in trange(50):
                market.simulate_timestep()

            exp = market.export()

            period_rewards = exp.agent_reward_history.reshape((n_agents, PERIOD, -1)).sum(axis=1)

            period_e = [gini(np.array(r)) for r in period_rewards if sum(r) > 0]
            e = [gini(np.array(r)) for r in exp.agent_total_rewards]
            all_r.append([job_task_r, market_limit, np.mean(e), np.std(e), np.std(e) / len(e), np.mean(period_e), np.std(period_e), np.std(period_e) / len(period_e)])
# %%
import pandas as pd
gini_df = pd.DataFrame(all_r, columns=["ratio", "parallel", "mean_e", "std_e", "sem_e", "mean_pe", "std_pe", "sem_pe"])
gini_df

# %% ============= Okun's Law Experiments ==================

n_tasks = 5
n_jobs = 1
n_agents = 50
market_limit = 1
market_pref_limit = 5
agent_pref_limit = 5
job_p = 0.8
train_p = 0.2
n_steps = 100

PERIOD = 5

okun_unemployed = []
okun_unemployed_total = []
okun_gdp = []
okun_all_unemployed = []
okun_all_unfilled = []

for n_agents in np.arange(30, 51, 4):
    for n_tasks in np.arange(30, 51, 4):
        task_ids = [f"task_{i}" for i in range(n_tasks)]

        tasks = [ProxyTask(t, noise=0.2) for t in task_ids]

        jobs = [
            Job(
                id=f"{task_id}_{i}",
                task_id=task_id,
                job_p=np.clip(job_p + np.random.normal(0, 0.1), 0, 1),
                base_reward=10,
            )
            for task_id in task_ids
            for i in range(n_jobs)
        ]

        agents = [PolicyAgent(agent_id=f"pol_{i}", jobs=jobs, model=None, verbose=False) for i in range(n_agents)]

        for agent in agents:
            agent.set_policy(train_p=np.clip(train_p + np.random.normal(0, 0.1), 0, 1))
        for agent in agents[:40]:
            task_preferences = [agent.task_ids[i] for i in np.random.permutation(agent.n_tasks)]
            job_preferences = [agent.job_ids[i] for i in np.random.permutation(agent.n_jobs)]

            agent.set_policy(
                task_preferences=task_preferences,
                job_preferences=job_preferences,
                train_p=np.clip(train_p + np.random.normal(0, 0.1), 0, 1),
                underbid_factor=0.9,
            )

        market = LabourMarket(
            jobs=jobs,
            market_pref_limit=market_pref_limit,
            agent_pref_limit=agent_pref_limit,
            market_limit=market_limit,
            tasks=tasks,
            agents=agents,
            skill_phi=0.1,
            rep_window=50,
            rep_lambda=0.5,
            rep_sensitivity=2,
            gumbel_t=0.001,
        )

        for _ in trange(50):
            market.simulate_timestep()

        exp = market.export()

        count = 0
        unmatched_agents = 0
        unmatched_jobs = 0
        total_labor_force = 0
        unmatched_agent_rate = []
        unmatched_job_rate = []

        for hx in exp.history:
            count += 1
            labor_force = n_agents - np.sum([a.action == "train" for a in hx.agent_actions])

            unmatched_agent = len(hx.unmatched_agents) - np.sum([a.action == "train" for a in hx.agent_actions])
            unmatched_job = len(hx.unmatched_jobs) 

            unmatched_agents += unmatched_agent
            unmatched_jobs += unmatched_job
            total_labor_force += labor_force

            unmatched_agent_rate.append(unmatched_agent / labor_force)
            unmatched_job_rate.append(unmatched_job / labor_force)

            okun_all_unemployed.append(unmatched_agent / labor_force)
            okun_all_unfilled.append(unmatched_job / labor_force)
            okun_gdp.append(np.sum([t[1] for t in hx.job_performance.values()]))
            
            if count % PERIOD == 0:
                
                okun_unemployed_total.append((unmatched_agents / total_labor_force, unmatched_jobs / total_labor_force))
                okun_unemployed.append((np.mean(unmatched_agent_rate), np.mean(unmatched_job_rate)))

                unmatched_agents = 0
                unmatched_jobs = 0
                total_labor_force = 0
                unmatched_agent_rate = []
                unmatched_job_rate = []

# %%
import numpy as np
from scipy import stats
from scipy.stats import zscore
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score


def remove_outliers_residual_based(X, y, slope, intercept, outlier_percent=5):
    """Remove outliers based on residuals from initial linear fit"""
    y_pred = slope * X + intercept
    residuals = np.abs(y - y_pred)
    threshold = np.percentile(residuals, 100 - outlier_percent)
    mask = residuals <= threshold
    return X[mask], y[mask], mask


def remove_outliers_zscore(X, y, z_threshold=2.5):
    """Remove outliers based on Z-score"""
    z_scores_x = np.abs(zscore(X))
    z_scores_y = np.abs(zscore(y))
    mask = (z_scores_x < z_threshold) & (z_scores_y < z_threshold)
    return X[mask], y[mask], mask


def remove_outliers_cook_distance(X, y, threshold=4):
    """Remove outliers based on Cook's distance"""
    from sklearn.linear_model import LinearRegression

    X_reshaped = X.reshape(-1, 1) if X.ndim == 1 else X
    model = LinearRegression().fit(X_reshaped, y)
    y_pred = model.predict(X_reshaped)

    # Calculate leverage (hat values)
    H = X_reshaped @ np.linalg.inv(X_reshaped.T @ X_reshaped) @ X_reshaped.T
    h = np.diag(H)

    # Calculate Cook's distance
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    p = X_reshaped.shape[1] + 1  # number of parameters
    n = len(y)

    cook_d = (residuals**2 / (p * mse)) * (h / (1 - h) ** 2)

    # Remove points with Cook's distance > threshold/n
    mask = cook_d <= threshold / n
    return X[mask], y[mask], mask


def linear_regression_with_outlier_removal(data, outlier_percent=5, method="residual", robust_method=None):
    """
    Perform linear regression with outlier removal

    Parameters:
    - data: numpy array of shape (n, 2) or tuple (X, y)
    - outlier_percent: percentage of outliers to remove
    - method: 'residual', 'zscore', or 'cook'
    - robust_method: None, 'huber', or 'theil_sen' for robust regression
    """

    # Handle input format
    if isinstance(data, tuple):
        X, y = data
    else:
        X, y = data[:, 0], data[:, 1]

    X = np.asarray(X)
    y = np.asarray(y)

    # Initial fit for outlier detection
    if method == "residual":
        slope_init, intercept_init, r_init, p_value_init, std_err_init = stats.linregress(X, y)
        X_clean, y_clean, mask = remove_outliers_residual_based(X, y, slope_init, intercept_init, outlier_percent)
    elif method == "zscore":
        z_threshold = np.sqrt(2 * np.log(100 / outlier_percent))
        X_clean, y_clean, mask = remove_outliers_zscore(X, y, z_threshold)
    elif method == "cook":
        X_clean, y_clean, mask = remove_outliers_cook_distance(X, y)
    else:
        raise ValueError("Method must be 'residual', 'zscore', or 'cook'")

    # Final regression on cleaned data
    if robust_method == "huber":
        # Huber regression (robust to outliers)
        X_reshaped = X_clean.reshape(-1, 1)
        huber = HuberRegressor(epsilon=1.35, max_iter=100)
        huber.fit(X_reshaped, y_clean)
        slope = huber.coef_[0]
        intercept = huber.intercept_

        # Calculate stats manually for robust regression
        y_pred = slope * X_clean + intercept
        r_value = np.corrcoef(y_clean, y_pred)[0, 1]
        r_squared = r2_score(y_clean, y_pred)

        # Standard error approximation for robust regression
        residuals = y_clean - y_pred
        mse = np.mean(residuals**2)
        x_var = np.var(X_clean)
        slope_std_err = np.sqrt(mse / (len(X_clean) * x_var))
        intercept_std_err = np.sqrt(mse * (1 / len(X_clean) + np.mean(X_clean) ** 2 / (len(X_clean) * x_var)))

        # P-value approximation (less reliable for robust regression)
        t_stat_slope = slope / slope_std_err
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat_slope), len(X_clean) - 2))

    elif robust_method == "theil_sen":
        # Theil-Sen estimator
        slope, intercept, low_slope, high_slope = stats.theilslopes(y_clean, X_clean)
        y_pred = slope * X_clean + intercept
        r_value = np.corrcoef(y_clean, y_pred)[0, 1]
        r_squared = r2_score(y_clean, y_pred)

        # Standard errors not directly available for Theil-Sen
        slope_std_err = (high_slope - low_slope) / (2 * 1.96)  # Approximation
        intercept_std_err = np.nan  # Not easily computed
        p_value = np.nan  # Not easily computed

    else:
        # Standard least squares regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_clean, y_clean)
        slope_std_err = std_err

        # Calculate intercept standard error
        x_mean = np.mean(X_clean)
        x_var = np.var(X_clean)
        intercept_std_err = std_err * np.sqrt((1 / len(X_clean)) + (x_mean**2) / (len(X_clean) * x_var))

    # Additional metrics
    y_pred_clean = slope * X_clean + intercept
    rmse = np.sqrt(np.mean((y_clean - y_pred_clean) ** 2))
    mae = np.mean(np.abs(y_clean - y_pred_clean))

    # Confidence intervals (95%)
    t_critical = stats.t.ppf(0.975, len(X_clean) - 2)
    slope_ci = [slope - t_critical * slope_std_err, slope + t_critical * slope_std_err]
    intercept_ci = [intercept - t_critical * intercept_std_err, intercept + t_critical * intercept_std_err]

    results = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,  # Correlation coefficient
        "r_squared": r_value**2 if not robust_method else r_squared,  # Coefficient of determination
        "p_value": p_value,
        "slope_std_err": slope_std_err,
        "intercept_std_err": intercept_std_err,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "rmse": rmse,
        "mae": mae,
        "X_clean": X_clean,
        "y_clean": y_clean,
        "outlier_mask": mask,
        "outliers_removed": len(X) - len(X_clean),
        "n_samples": len(X_clean),
    }

    return results


def plot_regression_results(X_orig, y_orig, results, title="Linear Regression with Outlier Removal"):
    """Plot the regression results"""
    plt.figure(figsize=(12, 10))

    # # Original data
    # outlier_mask = ~results['outlier_mask']
    # if np.any(outlier_mask):
    #     plt.scatter(X_orig[outlier_mask], y_orig[outlier_mask],
    #                color='red', alpha=0.6, s=50, label='Outliers (removed)', marker='x')

    # Clean data
    plt.scatter(results["X_clean"], results["y_clean"], color="blue", alpha=0.7, s=30)

    # Regression line
    X_line = np.linspace(np.min(X_orig), np.max(X_orig), 100)
    y_line = results["slope"] * X_line + results["intercept"]
    plt.plot(X_line, y_line, "r-", linewidth=2, label=f'y = {results["slope"]:.3f}x + {results["intercept"]:.3f}')

    # Confidence interval for the line
    y_pred_clean = results["slope"] * results["X_clean"] + results["intercept"]
    residuals = results["y_clean"] - y_pred_clean
    mse = np.mean(residuals**2)
    x_mean = np.mean(results["X_clean"])
    x_var = np.var(results["X_clean"])

    # Standard error of prediction
    se_line = np.sqrt(mse * (1 / len(results["X_clean"]) + (X_line - x_mean) ** 2 / (len(results["X_clean"]) * x_var)))
    t_critical = stats.t.ppf(0.975, len(results["X_clean"]) - 2)
    y_ci_lower = y_line - t_critical * se_line
    y_ci_upper = y_line + t_critical * se_line

    plt.fill_between(X_line, y_ci_lower, y_ci_upper, alpha=0.2, color="red", label="95% Confidence Interval")

    plt.xlabel("Change in unemployment rate")
    plt.ylabel("Change in GDP Growth")
    plt.title("")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f'R = {results["r_value"]:.4f}\n'
    stats_text += f'R² = {results["r_squared"]:.4f}\n'
    stats_text += f'RMSE = {results["rmse"]:.4f}\n'
    stats_text += f'p-value = {results["p_value"]:.2e}\n' if not np.isnan(results["p_value"]) else "p-value = N/A\n"
    stats_text += f'n = {results["n_samples"]}\n'
    # stats_text += f'Outliers removed = {results["outliers_removed"]}'

    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()
# %%
# Beveridge data

n_tasks = 5
n_jobs = 10
n_agents = 50
market_limit = 1
market_pref_limit = 5
agent_pref_limit = 5
job_p = 0.8
train_p = 0.2
n_steps = 100

PERIOD = 5

unemployed_beveridge = []
unemployed_total_beveridge = []

for n_agents in np.arange(30, 51, 4):
    for n_tasks in np.arange(30, 51, 4):
        n_jobs = 1
        task_ids = [f"task_{i}" for i in range(n_tasks)]

        tasks = [ProxyTask(t, noise=0.1) for t in task_ids]

        jobs = [
            Job(
                id=f"{task_id}_{i}",
                task_id=task_id,
                job_p=np.clip(job_p + np.random.normal(0, 0.1), 0, 1),
                base_reward=10,
            )
            for task_id in task_ids
            for i in range(n_jobs)
        ]

        agents = [PolicyAgent(agent_id=f"pol_{i}", jobs=jobs, model=None, verbose=False) for i in range(n_agents)]

        for agent in agents:
            agent.set_policy(train_p=np.clip(train_p + np.random.normal(0, 0.1), 0, 1))
        for agent in agents[:40]:
            task_preferences = [agent.task_ids[i] for i in np.random.permutation(agent.n_tasks)]
            job_preferences = [agent.job_ids[i] for i in np.random.permutation(agent.n_jobs)]

            agent.set_policy(
                task_preferences=task_preferences,
                job_preferences=job_preferences,
                train_p=np.clip(train_p + np.random.normal(0, 0.1), 0, 1),
                underbid_factor=0.9,
            )

        market = LabourMarket(
            jobs=jobs,
            market_pref_limit=market_pref_limit,
            agent_pref_limit=agent_pref_limit,
            market_limit=market_limit,
            tasks=tasks,
            agents=agents,
            skill_phi=0.1,
            rep_window=50,
            rep_lambda=0.5,
            rep_sensitivity=2,
            gumbel_t=0.001,
        )

        for _ in trange(50):
            market.simulate_timestep()

        exp = market.export()

        count = 0
        unmatched_agents = 0
        unmatched_jobs = 0
        total_labor_force = 0
        unmatched_agent_rate = []
        unmatched_job_rate = []

        for hx in exp.history:
            count += 1
            labor_force = n_agents - np.sum([a.action == "train" for a in hx.agent_actions])

            unmatched_agent = len(hx.unmatched_agents) - np.sum([a.action == "train" for a in hx.agent_actions])
            unmatched_job = len(hx.unmatched_jobs) 

            unmatched_agents += unmatched_agent
            unmatched_jobs += unmatched_job
            total_labor_force += labor_force

            unmatched_agent_rate.append(unmatched_agent / labor_force)
            unmatched_job_rate.append(unmatched_job / labor_force)
            
            if count % PERIOD == 0:
                
                unemployed_total_beveridge.append((unmatched_agents / total_labor_force, unmatched_jobs / total_labor_force))
                unemployed_beveridge.append((np.mean(unmatched_agent_rate), np.mean(unmatched_job_rate)))

                unmatched_agents = 0
                unmatched_jobs = 0
                total_labor_force = 0
                unmatched_agent_rate = []
                unmatched_job_rate = []

# %%
u = np.array(unemployed_beveridge)
# %%
# plt.scatter(u[:, 0], u[:, 1])# %%

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
ax = axes[1]

# Fit with outlier removal
results = fit_hyperbola_with_outlier_removal(u, outlier_percent=5, method="residual")

print(f"Fitted parameters [a, b, c]: {results['params']}")
print(f"R²: {results['r_squared']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"Outliers removed: {results['outliers_removed']}")

# Plot results
X = u[:, 0]
y = u[:, 1]
# plt.scatter(X, y, label='Original data', s=10)
ax.scatter(results["X_clean"], results["y_clean"], color=SNS_GREY, alpha=0.8, s=20)

X_plot = np.linspace(0.01, 0.3, 1000)
y_plot = hyperbolic_func(X_plot, *results["params"])
ax.plot(
    X_plot,
    y_plot,
    color=SNS_RED,
    linewidth=4,
    label=f"R²={results['r_squared']:.3f}"
)

ax.set_xlabel("Unemployment Rate", fontsize=20)
ax.set_ylabel("Job Vacancy Rate", fontsize=20)
ax.tick_params(labelsize=18)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_xlim(-0.01, 0.33)
ax.set_ylim(-0.01, 0.55)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=15)
#### OKUN 
# PERIOD = 5

_gdp = np.array(okun_gdp).reshape((36, PERIOD, -1)).sum(axis=1)
gdp_diff = (np.diff(_gdp) / _gdp[:, :-1]).flatten() * 100
gdp_diff.shape
_unemploymnet = np.array(okun_all_unemployed).reshape((36, PERIOD, -1)).mean(axis=1)
# _unemploymnet = np.array([u[0] for u in unemployed_total]).reshape((36, -1))
u_diff = np.diff(_unemploymnet).flatten() * 100

# plt.scatter(u_diff, gdp_diff)
np.random.seed(42)

y_true = gdp_diff
X_true = u_diff

# Different methods comparison
methods = ["residual", "zscore", "cook"]
robust_methods = [None, "huber"]

method = "residual"
robust = "huber"
method_name = f"{method}" + (f" + {robust}" if robust else "")
print(f"\n=== Method: {method_name} ===")

results = linear_regression_with_outlier_removal(
    (X_true, y_true), outlier_percent=5, method="zscore", robust_method=robust
)

print(f"Slope: {results['slope']:.4f} ± {results['slope_std_err']:.4f}")
print(f"Intercept: {results['intercept']:.4f} ± {results['intercept_std_err']:.4f}")
print(f"R-value: {results['r_value']:.4f}")
print(f"R²: {results['r_squared']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"p-value: {results['p_value']:.2e}" if not np.isnan(results["p_value"]) else "p-value: N/A")
print(f"Outliers removed: {results['outliers_removed']}")

ax = axes[0]
# Clean data
ax.scatter(results["X_clean"], results["y_clean"], color=SNS_GREY, alpha=0.7, s=30)

# Regression line
X_line = np.linspace(np.min(X_true), np.max(X_true), 100)
y_line = results["slope"] * X_line + results["intercept"]
ax.plot(X_line, y_line, color=SNS_RED, linewidth=4, label=f"R²={results['r_squared']:.3f}, β={results['slope']:.2f}")

# Confidence interval for the line
y_pred_clean = results["slope"] * results["X_clean"] + results["intercept"]
residuals = results["y_clean"] - y_pred_clean
mse = np.mean(residuals**2)
x_mean = np.mean(results["X_clean"])
x_var = np.var(results["X_clean"])


ax.set_xlabel("Change in unemployment rate (%)", fontsize=20)
ax.set_ylabel("Change in GDP Growth (%)", fontsize=20)
ax.set_title("")
ax.set_yticks((-20, -10, 0, 10, 20))
ax.tick_params(labelsize=18)
ax.legend(fontsize=15)
ax.grid(True, alpha=0.3)

# Add statistics text box
stats_text = f'R = {results["r_value"]:.4f}\n'
stats_text += f'R² = {results["r_squared"]:.4f}\n'
stats_text += f'RMSE = {results["rmse"]:.4f}\n'
stats_text += f'p-value = {results["p_value"]:.2e}\n' if not np.isnan(results["p_value"]) else "p-value = N/A\n"
stats_text += f'n = {results["n_samples"]}\n'
# stats_text += f'Outliers removed = {results["outliers_removed"]}'

# ax.text(
#     0.05,
#     0.95,
#     stats_text,
#     transform=plt.gca().transAxes,
#     verticalalignment="top",
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
# )

ax = axes[2]
for ix, ratio in enumerate([1, 4, 16, 64][::-1]):
    _df = gini_df.query('ratio == @ratio').groupby('parallel').max().reset_index()
    ax.plot(_df.parallel, _df.mean_pe, linewidth=4, color=sns.color_palette("viridis")[5 - ix], label=f'B={64 // ratio}')
    
    last = _df.mean_pe.iloc[-1]
    ax.text(8.1, last, f"{last:.2f}", fontsize=18, va='center', color='dimgrey')



ax.set_ylabel("Gini Coefficient", fontsize=20)
ax.set_xlabel("Concurrent Job Capacity", fontsize=20)
ax.set_yticks((0.2, 0.4, 0.6))
ax.tick_params(axis='both', which='major', labelsize=18)
ax.legend(fontsize=15)
ax.set_xlim(0.8, 9.5)
ax.set_ylim(0.1, 0.75)
ax.grid(True, alpha=0.3)

plt.text(-0.1, 1.02, 'A', ha='left', va='top', fontsize=25, weight='bold', transform=axes[0].
         transAxes)
plt.text(-0.1, 1.02, 'B', ha='left', va='top', fontsize=25, weight='bold', transform=axes[1].
         transAxes)
plt.text(-0.1, 1.02, 'C', ha='left', va='top', fontsize=25, weight='bold', transform=axes[2].
         transAxes)

plt.tight_layout()
plt.show()


# %%

# %%
