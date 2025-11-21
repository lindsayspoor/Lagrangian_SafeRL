

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from typing import List, Optional
import argparse


def plot_training_progress(log_dirs: Optional[List[str]] = None,
    save_dir: str = '',
    env: str = '',
    algo: str = '',
    exp: str = '',
    cost_lim: float = 25.0,
    lambda_init: float = 0.03,
) -> None:
    

    all_rewards = []
    all_costs = []
    min_length = float('inf')

    # in case not all models are trained on an equal amount of training iterations, the training curves are plotted
    # on the range of the model that has the shortest length ofn total training iterations.

    # collect data and track shortest length
    for log_dir in log_dirs:
        df = pd.read_csv(f"{log_dir}/progress.csv")

        if 'Metrics/LagrangeMultiplier' not in df.columns:
            raise ValueError(f"Lagrange multiplier data not found in {log_dir}")

        rewards = df['Metrics/EpRet'].values
        costs = df['Metrics/EpCost'].values

        min_length = min(min_length, len(rewards), len(costs))  # Track shortest length

        all_rewards.append(rewards)
        all_costs.append(costs)

    # trim all arrays to the shortest length
    all_rewards_trimmed = [r[:min_length] for r in all_rewards]
    all_costs_trimmed = [c[:min_length] for c in all_costs]

    # Convert to NumPy arrays
    rewards = np.array(all_rewards_trimmed)
    costs = np.array(all_costs_trimmed)

    # Average across seeds
    avg_reward = np.mean(rewards, axis=0)
    avg_cost = np.mean(costs, axis=0)


    std_curve_reward = np.std(np.array(all_rewards_trimmed), axis=0)
    std_curve_cost = np.std(np.array(all_costs_trimmed), axis=0)
    
    epochs = np.arange(len(avg_reward))

    sns.set(style='whitegrid', context='paper', font_scale=1.6)
    sns.set_palette('deep')

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    color_reward = 'mediumblue'
    ax1.set_ylabel('Reward', fontsize=18, color=color_reward)

    ax1.plot(epochs * 20000, avg_reward, linewidth=1.5,
            color=color_reward, label='Reward')
    ax1.fill_between(
        epochs * 20000,
        avg_reward - std_curve_reward,
        avg_reward + std_curve_reward,
        alpha=0.2,
        color=color_reward
    )

    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.tick_params(labelsize=16)
    ax1.grid(True)


    color_cost = 'crimson'
    ax2.set_xlabel('Timesteps', fontsize=18)
    ax2.set_ylabel('Cost', fontsize=18, color=color_cost)

    ax2.plot(epochs * 20000, avg_cost, linewidth=1.5,
            color=color_cost, label='Cost')
    ax2.fill_between(
        epochs * 20000,
        avg_cost - std_curve_cost,
        avg_cost + std_curve_cost,
        alpha=0.2,
        color=color_cost
    )

    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.tick_params(labelsize=16)
    ax2.grid(True)

    fig.suptitle(env, fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


    if exp == 'fixed_lambda':
        plt.savefig(f"{save_dir}/training_curves_{exp}_{algo}_{env}_{cost_lim=}_{lambda_init=}.pdf")
    if exp == 'auto_update_GA':
        plt.savefig(f"{save_dir}/training_curves_{exp}_{algo}_{env}_{cost_lim=}.pdf")
    
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SafetyPointCircle1-v0')
    parser.add_argument('--algo', type=str, default='PPOLag')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--cost_lim', type=float, default=25.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[])
    parser.add_argument('--timesteps', type=int, default=10000000)
    parser.add_argument('--exp', type=str, default='fixed_lambda')
    parser.add_argument('--lambda_values', type=float, nargs='+', default = [])


    return parser.parse_args()

def main():

    log_dirs = []

    args = parse_args()

    algo = args.algo
    env = args.env
    cost_lim = args.cost_lim
    seeds = args.seeds
    timesteps = args.timesteps
    save_dir = args.save_dir
    exp = args.exp
    lambda_values = args.lambda_values


    if exp == 'fixed_lambda':
        for lambda_init in lambda_values:
            log_dirs = []
            for seed in seeds:
                log_dir = f'./exp-x/{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}_{lambda_init=}/seed_{seed}'
                log_dirs.append(log_dir)
            plot_training_progress(log_dirs = log_dirs, save_dir = save_dir, env = env, algo = algo, exp=exp, cost_lim=cost_lim, lambda_init = lambda_init)


    if exp == 'auto_update_GA':
        if algo != "PPOLag":
            raise ValueError(f"Invalid algo '{algo}'. Expected 'PPOLag'.")
        for seed in seeds:
            log_dir = f'./exp-x/{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}/seed_{seed}'
            log_dirs.append(log_dir)

        plot_training_progress(log_dirs = log_dirs, save_dir = save_dir, env = env, algo = algo, exp=exp, cost_lim=cost_lim)




if __name__ == '__main__':
    main()



