

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import List
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess
import glob
from matplotlib.ticker import ScalarFormatter


def plot_pareto_frontier(
    lambda_values,
    avg_reward, avg_cost,
    std_reward=None, std_cost=None,
    save_dir='',
    env_id='',
    algo='',
    cost_lim=25.0,
    smooth=False,
    n_seeds=10,                 # used for 95% CI band on frontier (SEM = SD/sqrt(n))
    show_point_errorbars=True,  # datapoints: ±1 SD
    show_frontier_band=True,    # frontier: 95% CI of mean
):
    """
    Plot return as a function of cost for all lambda values.

    Conventions:
      - Datapoint error bars show ±1 standard deviation across seeds (x=cost, y=return).
      - Pareto frontier shaded region shows the 95% confidence interval of the mean
        (computed as ±1.96 * SEM, with SEM = SD/sqrt(n_seeds)), using std_reward.

    Note: If std_reward/std_cost are smoothed (e.g., LOWESS), small negative values can appear.
          We clip std arrays to be nonnegative before plotting (Matplotlib requires this).
    """


    # --- Convert to arrays ---
    lam = np.asarray(lambda_values, dtype=float)
    r = np.asarray(avg_reward, dtype=float)
    c = np.asarray(avg_cost, dtype=float)

    sr = None if std_reward is None else np.asarray(std_reward, dtype=float)
    scost = None if std_cost is None else np.asarray(std_cost, dtype=float)

    # --- Validity mask (keep stds aligned) ---
    valid = np.isfinite(lam) & np.isfinite(r) & np.isfinite(c)
    if sr is not None:
        valid &= np.isfinite(sr)
    if scost is not None:
        valid &= np.isfinite(scost)

    lam, r, c = lam[valid], r[valid], c[valid]
    if sr is not None:
        sr = sr[valid]
    if scost is not None:
        scost = scost[valid]

    # --- Sort by cost (natural for frontier curve) ---
    order = np.argsort(c)
    lam, r, c = lam[order], r[order], c[order]
    if sr is not None:
        sr = sr[order]
    if scost is not None:
        scost = scost[order]

    # IMPORTANT: clip stds to be nonnegative (LOWESS smoothing can go slightly negative)
    if sr is not None:
        sr = np.clip(sr, 0.0, None)
    if scost is not None:
        scost = np.clip(scost, 0.0, None)

    # --- Optional smoothing in cost-space (for frontier detection only) ---
    if smooth and len(c) >= 5 and lowess is not None:
        r_s = lowess(r, c, frac=0.4, it=0, return_sorted=False)
    else:
        r_s = r

    # --- Build empirical Pareto frontier: upper envelope in (cost, return) ---
    running_max = -np.inf
    frontier_idx = []
    for i in range(len(c)):
        if r_s[i] > running_max:
            running_max = r_s[i]
            frontier_idx.append(i)
    frontier_idx = np.array(frontier_idx, dtype=int)

    # --- Plot styling ---
    sns.set(style='white', context='paper', font_scale=1.6)
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Scatter all achieved points, colored by log10(lambda) ---
    sc = ax.scatter(
        c, r,
        c=np.log10(lam),
        s=100,
        alpha=0.9,
        edgecolor='black',
        zorder=3,
        cmap='PuRd',
    )

    # --- Datapoint error bars: ±1 SD (across seeds) ---
    if show_point_errorbars and (sr is not None or scost is not None):
        xerr = None if scost is None else scost
        yerr = None if sr is None else sr

        # extra safety: matplotlib requires nonnegative
        if xerr is not None:
            xerr = np.clip(xerr, 0.0, None)
        if yerr is not None:
            yerr = np.clip(yerr, 0.0, None)

        ax.errorbar(
            c, r,
            xerr=xerr,
            yerr=yerr,
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=2.5,
            alpha=0.35,
            zorder=2,
        )

    # --- Frontier curve ---
    ax.plot(
        c[frontier_idx],
        r[frontier_idx],
        linewidth=2.5,
        zorder=4,
        linestyle='--',
        color='black',
    )

    # --- Frontier shaded region: 95% CI of mean return (±1.96 * SEM) ---
    if show_frontier_band and (sr is not None) and len(frontier_idx) > 1:
        if n_seeds is None or n_seeds <= 1:
            raise ValueError("n_seeds must be > 1 to compute a 95% CI band from std over seeds.")

        cf = c[frontier_idx]
        rf = r[frontier_idx]
        srf = sr[frontier_idx]

        srf = np.clip(srf, 0.0, None)  # safety
        sem = srf / np.sqrt(float(n_seeds))

        lower = rf - 1.96 * sem
        upper = rf + 1.96 * sem

        ax.fill_between(
            cf,
            lower,
            upper,
            alpha=0.18,
            zorder=1,
            color='black',
            linewidth=0,
        )

    ax.set_xlabel("Cost", fontsize=24)
    ax.set_ylabel("Return", fontsize=24)
    ax.set_title(f"{env_id}", fontsize=24)

    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter_y)
    ax.yaxis.get_offset_text().set_fontsize(20)

    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter_x)
    ax.xaxis.get_offset_text().set_fontsize(20)

    # --- Colorbar for lambda ---
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(\lambda)$", fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    ax.tick_params(labelsize=24)
    sns.despine(ax=ax)
    fig.tight_layout()

    # --- Save ---
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = f"{save_dir}/{algo}_{env_id}_pareto_frontier.pdf"
    else:
        out_path = f"{algo}_{env_id}_pareto_frontier.pdf"


    plt.savefig(out_path, bbox_inches="tight")
    plt.close()



def load_lambda_data(
    log_dirs: List[str],
    lambda_values: list,
    seeds: list,
) -> tuple:
    """
    Load training data for all seeds and lambda values, compute per-seed
    averages over the last 88 episodes, then smooth with LOWESS.

    Returns:
        x        – sorted lambda values (np.ndarray)
        r_s      – smoothed mean return (np.ndarray)
        c_s      – smoothed mean cost (np.ndarray)
        r_sd_s   – smoothed std of return (np.ndarray)
        c_sd_s   – smoothed std of cost (np.ndarray)
    """
    rewards_all_seeds = []
    costs_all_seeds = []

    for seed_num in seeds:
        seed = 'seed_' + str(seed_num)
        all_rewards = []
        all_costs = []

        for log_dir in log_dirs:
            df = pd.read_csv(f"{log_dir}/{seed}/progress.csv")
            all_rewards.append(df['Metrics/EpRet'].iloc[-88:].mean())
            all_costs.append(df['Metrics/EpCost'].iloc[-88:].mean())

        rewards_all_seeds.append(all_rewards)
        costs_all_seeds.append(all_costs)

    avg_reward = np.mean(np.array(rewards_all_seeds), axis=0)
    avg_cost = np.mean(np.array(costs_all_seeds), axis=0)
    std_curve_reward = np.std(np.array(rewards_all_seeds), axis=0)
    std_curve_cost = np.std(np.array(costs_all_seeds), axis=0)

    # ensure sorted by lambda
    order = np.argsort(lambda_values)
    x = np.asarray(lambda_values)[order]
    r = np.asarray(avg_reward)[order]
    c = np.asarray(avg_cost)[order]
    r_sd = np.asarray(std_curve_reward)[order]
    c_sd = np.asarray(std_curve_cost)[order]

    # frac is the smoothing strength (span); try between 0.4–0.6
    r_s    = lowess(r,    x, frac=0.5, it=0, return_sorted=False)
    c_s    = lowess(c,    x, frac=0.5, it=0, return_sorted=False)
    r_sd_s = lowess(r_sd, x, frac=0.5, it=0, return_sorted=False)
    c_sd_s = lowess(c_sd, x, frac=0.5, it=0, return_sorted=False)

    return x, r_s, c_s, r_sd_s, c_sd_s


def plot_lambda_profile(
    x,
    r_s,
    c_s,
    r_sd_s,
    c_sd_s,
    save_dir: str = '',
    env_id: str = '',
    algo: str = '',
    cost_lim: float = 25.0,
) -> None:

    # plotting the results
    sns.set(style='white', context='paper', font_scale=1.6) 
    sns.set_palette('deep')  # Optional: color palettes

    
    # Set up the figure and first axis
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xscale('log')
    # ax1.semilogx()
    # Plot reward on the left Y-axis
    color_reward = 'midnightblue'
    ax1.set_xlabel(r'$\lambda$', fontsize=25)
    ax1.set_ylabel('Return', fontsize=25, color=color_reward)
    ax1.plot(x, r_s, linewidth=2.4, color=color_reward, label='Reward')
    ax1.fill_between(x, r_s - 1.96*r_sd_s, r_s + 1.96*r_sd_s, alpha=0.1, color=color_reward, linewidth=0)
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.set_xlim(left=x[0], right=x[-1])


    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-3, 3))  # always use scientific notation
    

    ax1.yaxis.set_major_formatter(formatter_y)
    offset1 = ax1.yaxis.get_offset_text()
    offset1.set_fontsize(20)

    # Move it next to the axis instead of above
    
    offset1.set_y(1.0)     # vertical position (1.0 = top of axis)


    ax2 = ax1.twinx()
    # Plot cost on the right Y-axis
    color_cost = 'crimson'
    ax2.set_ylabel('Cost', fontsize=24, color=color_cost)
    ax2.plot(x, c_s, linewidth=2.4, color=color_cost, label='Cost')
    ax2.fill_between(x, c_s - 1.96*c_sd_s, c_s + 1.96*c_sd_s, alpha=0.1, color=color_cost, linewidth=0)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.hlines(y=25.0, xmin=min(x), xmax=max(x),linestyles='dashed', color='black', alpha=1, zorder=50, linewidth=1.5)


    plt.title(env_id, fontsize=22, y=0.998, x=0.57)
    ax1.tick_params(labelsize=24)
    ax2.tick_params(labelsize=24)


    formatter_y2 = ScalarFormatter(useMathText=True)
    formatter_y2.set_scientific(True)
    formatter_y2.set_powerlimits((-3, 3))  # always use scientific notation
    ax2.yaxis.set_major_formatter(formatter_y2)
    offset2 = ax2.yaxis.get_offset_text()
    offset2.set_fontsize(20)
    # Move it next to the right axis
    offset2.set_x(1.15)    # slightly right of axis
    offset2.set_y(1.0)
    # ax1.grid(True)
    fig.tight_layout(rect=[0, 0, 1, 1], pad=0.1)

    # Save
    plt.savefig(f"{save_dir}/{algo}_{env_id}_lambda_profile.pdf")







def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_values', type=float, nargs='+', default=[])
    parser.add_argument('--algo', type=str, default='PPOLag')
    parser.add_argument('--env', type=str, default='SafetyPointCircle1-v0')
    parser.add_argument('--cost_lim', type=float, default=25.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[])
    parser.add_argument('--timesteps', type=int, default=10000000)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--exp', type=str, default = 'fixed_lambda')


    return parser.parse_args()


def main():

    log_dirs = []
    args = parse_args()
    lambda_values = args.lambda_values
    algo = args.algo
    env = args.env
    cost_lim = args.cost_lim
    seeds = args.seeds
    timesteps = args.timesteps
    save_dir = args.save_dir
    exp = args.exp


    for lambda_init in lambda_values:
        pattern = f'./exp-x/*{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}_{lambda_init=}/'
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No runs match pattern:\n{pattern}")

        log_dir = matches[0]   # or loop over all matches
        log_dirs.append(log_dir)

    x, r_s, c_s, r_sd_s, c_sd_s = load_lambda_data(log_dirs=log_dirs, lambda_values=lambda_values, seeds=seeds)
    plot_lambda_profile(x=x, r_s=r_s, c_s=c_s, r_sd_s=r_sd_s, c_sd_s=c_sd_s, save_dir=save_dir, env_id=env, algo=algo, cost_lim=cost_lim)
    plot_pareto_frontier(lambda_values=x, avg_reward=r_s, avg_cost=c_s, std_reward=r_sd_s, std_cost=c_sd_s, save_dir=save_dir, env_id=env, algo=algo, cost_lim=cost_lim, smooth=False)

    

if __name__ == '__main__':
    main()



