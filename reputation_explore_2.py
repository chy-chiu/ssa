# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ----------------------------
# Simulation: skill + noise in [0,1]
# ----------------------------
def simulate_q(T=200, seed=0, noise_sigma=0.12):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # Smooth sigmoid skill from 0 -> 1
    tau = T / 12.0
    skill = 1 / (1 + np.exp(-(t - T/2)/tau))
    noise = rng.normal(0, noise_sigma, size=T)
    q = np.clip(skill + noise, 0, 1)
    return t, skill, q

# ----------------------------
# Fuzzy triangular membership for k bins on [0,1]
# Each triangle is centered at c_i = (i-1)/(k-1), base width 2/(k-1)
# r_i(q) = max(1 - (k-1)*|q - c_i|, 0)
# ----------------------------
def triangular_membership(q, k):
    centers = np.linspace(0, 1, k)
    r = np.maximum(1.0 - (k - 1) * np.abs(q - centers), 0.0)
    # r sums to 1 for q in [0,1] given this construction (up to float error)
    # For safety, renormalize tiny numerical drift:
    s = r.sum()
    if s > 0:
        r /= s
    return r

# ----------------------------
# Aggregation with sliding window W and exponential aging lambda_
# For time t, weight of observation at time s is lambda_^(t-s)
# ----------------------------
def aggregate_dirichlet(window, t, k=5, lambda_=0.95, C=2.0):
    # Accumulate weighted soft counts across the window
    counts = np.zeros(k)
    for (s, q_s) in window:
        w = lambda_ ** (t - s)
        counts += w * triangular_membership(q_s, k)
    # Dirichlet posterior expectation with base-rate a=uniform
    a = np.ones(k) / k
    S = (counts + C * a) / (C + counts.sum())
    # Point estimate on [0,1] by mapping levels to evenly spaced values
    v = np.linspace(0, 1, k)
    point_est = (S * v).sum()
    return S, point_est, counts

def aggregate_beta(window, t, lambda_=0.95, alpha0=1.0, beta0=1.0):
    # Fractional soft counts: alpha += w*q, beta += w*(1-q)
    alpha = alpha0
    beta = beta0
    for (s, q_s) in window:
        w = lambda_ ** (t - s)
        alpha += w * q_s
        beta  += w * (1 - q_s)
    mean = alpha / (alpha + beta)
    return mean, alpha, beta

# ----------------------------
# Run both systems on simulated data
# ----------------------------
def run(T=200, W=60, lambda_=0.95, k=5, C=2.0, seed=0, noise_sigma=0.12):
    t, skill, q = simulate_q(T=T, seed=seed, noise_sigma=noise_sigma)
    window = deque(maxlen=W)

    dirichlet_point = np.zeros(T)
    dirichlet_S = np.zeros((T, k))
    dirichlet_counts = np.zeros((T, k))

    beta_mean = np.zeros(T)
    alpha_traj = np.zeros(T)
    beta_traj = np.zeros(T)

    for i in range(T):
        window.append((i, q[i]))

        S, pe, counts = aggregate_dirichlet(window, i, k=k, lambda_=lambda_, C=C)
        dirichlet_point[i] = pe
        dirichlet_S[i] = S
        dirichlet_counts[i] = counts

        bm, alpha, beta = aggregate_beta(window, i, lambda_=lambda_, alpha0=C/2, beta0=C/2)
        beta_mean[i] = bm
        alpha_traj[i] = alpha
        beta_traj[i] = beta

    return dict(
        t=t, skill=skill, q=q,
        dirichlet_point=dirichlet_point, dirichlet_S=dirichlet_S, dirichlet_counts=dirichlet_counts,
        beta_mean=beta_mean, alpha=alpha_traj, beta=beta_traj
    )

# ----------------------------
# Polarization index for the 5-level distribution
# A simple signal: mass at extremes minus mass at center
# pol = (p1 + p5) - p3  in [ -1, 1 ]
# ----------------------------
def polarization_index(S5):
    return (S5[:,0] + S5[:,-1]) - S5[:, S5.shape[1]//2]

# ----------------------------
# Visualization
# ----------------------------
def plot_results(res, k=5):
    t = res['t']
    skill = res['skill']
    q = res['q']
    dirichlet_point = res['dirichlet_point']
    dirichlet_S = res['dirichlet_S']
    beta_mean = res['beta_mean']

    pol = polarization_index(dirichlet_S)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # 1) Signal
    axs[0].plot(t, skill, label='Skill (latent)', color='black', linewidth=2, alpha=0.6)
    axs[0].plot(t, q, '.', label='Observed performance q', color='tab:blue', markersize=3, alpha=0.5)
    axs[0].set_ylabel('Value')
    axs[0].set_title('Skill + noise (clipped to [0,1])')
    axs[0].legend()

    # 2) Reputation trajectories
    axs[1].plot(t, beta_mean, label='Beta mean (continuous updater)', color='tab:orange', linewidth=2)
    axs[1].plot(t, dirichlet_point, label='Dirichlet 5-star point estimate', color='tab:green', linewidth=2, alpha=0.8)
    axs[1].set_ylabel('Reputation (0..1)')
    axs[1].set_title('Reputation over time (window + aging)')
    axs[1].legend()

    # 3) Polarization from 5-star distribution
    axs[2].plot(t, pol, label='Polarization (p1+p5 - p3)', color='tab:red')
    axs[2].axhline(0, color='gray', linestyle='--', linewidth=1)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Polarization')
    axs[2].set_title('Polarization detectable only with multinomial representation')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    # Snapshot bar plots of 5-star multinomial S at selected times
    snap_ts = [int(0.2*len(t)), int(0.5*len(t)), int(0.8*len(t))]
    v = np.linspace(1, k, k)  # 1..5 stars
    fig2, ax2 = plt.subplots(1, len(snap_ts), figsize=(12, 3), sharey=True)
    for j, idx in enumerate(snap_ts):
        ax2[j].bar(v, dirichlet_S[idx], width=0.7, color='tab:green', alpha=0.8)
        ax2[j].set_xticks(v)
        ax2[j].set_ylim(0, 1)
        ax2[j].set_title(f'Time {idx}, q={q[idx]:.2f}\nDir point={res["dirichlet_point"][idx]:.2f}, Beta mean={res["beta_mean"][idx]:.2f}')
        ax2[j].set_xlabel('Stars')
        if j == 0:
            ax2[j].set_ylabel('Probability')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Hyperparams
    T = 240         # total time steps
    W = 60          # sliding window size (recent observations only)
    lambda_ = 0.95  # exponential aging
    k = 5           # 5-star system
    C = 2.0         # base-rate mass (same spirit as in the paper)
    seed = 1        # RNG seed
    noise_sigma = 0.12

    res = run(T=T, W=W, lambda_=lambda_, k=k, C=C, seed=seed, noise_sigma=noise_sigma)
    plot_results(res, k=k)

# %%
