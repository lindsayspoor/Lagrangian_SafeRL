

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from typing import List, Optional
import argparse


def plot_training_progress_reward_cost(log_dirs: Optional[List[str]] = None,
    save_dir: str = '',
    env: str = '',
    algo: str = '',
    exp: str = '',
    cost_lim: float = 25.0,
    lambda_init: float = 0.03,
    steps_epoch: int = 20000,
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

    ax1.plot(epochs * steps_epoch, avg_reward, linewidth=1.5,
            color=color_reward, label='Reward')
    ax1.fill_between(
        epochs * steps_epoch,
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

    ax2.plot(epochs * steps_epoch, avg_cost, linewidth=1.5,
            color=color_cost, label='Cost')
    ax2.fill_between(
        epochs * steps_epoch,
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



    plt.savefig(f"{save_dir}/training_curves_{exp}_{algo}_{env}_{cost_lim=}_{lambda_init=}.pdf")

    plt.close()


def plot_training_progress_lambda_reward_cost(log_dirs: Optional[List[str]] = None,
    save_dir: str = '',
    env: str = '',
    algo: str = '',
    exp: str = '',
    cost_lim: float = 25.0,
    k_p: float = 0.1,
    k_i: float= 0.1,
    k_d: float= 0.0,
    steps_epoch: int = 20000,
) -> None:
    

    all_rewards = []
    all_costs = []
    all_lambdas = []
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
        lambdas = df['Metrics/LagrangeMultiplier'].values

        min_length = min(min_length, len(rewards), len(costs), len(lambdas))  # Track shortest length

        all_rewards.append(rewards)
        all_costs.append(costs)
        all_lambdas.append(lambdas)

    # trim all arrays to the shortest length
    all_rewards_trimmed = [r[:min_length] for r in all_rewards]
    all_costs_trimmed = [c[:min_length] for c in all_costs]
    all_lambdas_trimmed = [l[:min_length] for l in all_lambdas]

    # Convert to NumPy arrays
    rewards = np.array(all_rewards_trimmed)
    costs = np.array(all_costs_trimmed)
    lambdas = np.array(all_lambdas)

    # Average across seeds
    avg_reward = np.mean(rewards, axis=0)
    avg_cost = np.mean(costs, axis=0)
    avg_lambdas = np.mean(lambdas, axis=0)


    std_curve_reward = np.std(np.array(all_rewards_trimmed), axis=0)
    std_curve_cost = np.std(np.array(all_costs_trimmed), axis=0)
    std_curve_lambdas = np.std(np.array(all_lambdas_trimmed), axis=0)

    print(f"{std_curve_lambdas=}")
    
    epochs = np.arange(len(avg_reward))

    sns.set(style='whitegrid', context='paper', font_scale=1.6)
    sns.set_palette('deep')

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    color_lambdas = 'indigo'
    ax1.set_ylabel(r"$\lambda$", fontsize=18, color=color_lambdas)

    ax1.plot(epochs * steps_epoch, avg_lambdas, linewidth=1.5,
            color=color_lambdas, label=r'$\lambda$')
    ax2.fill_between(
        epochs * steps_epoch,
        avg_lambdas - std_curve_lambdas,
        avg_lambdas + std_curve_lambdas,
        alpha=0.2,
        color=color_lambdas
    )

    ax1.tick_params(axis='y', labelcolor=color_lambdas)
    ax1.tick_params(labelsize=16)
    ax1.grid(True)

    color_reward = 'mediumblue'
    ax2.set_ylabel('Reward', fontsize=18, color=color_reward)

    ax2.plot(epochs * steps_epoch, avg_reward, linewidth=1.5,
            color=color_reward, label='Reward')
    ax2.fill_between(
        epochs * steps_epoch,
        avg_reward - std_curve_reward,
        avg_reward + std_curve_reward,
        alpha=0.2,
        color=color_reward
    )

    ax2.tick_params(axis='y', labelcolor=color_reward)
    ax2.tick_params(labelsize=16)
    ax2.grid(True)


    color_cost = 'crimson'
    ax3.set_xlabel('Timesteps', fontsize=18)
    ax3.set_ylabel('Cost', fontsize=18, color=color_cost)

    ax3.plot(epochs * steps_epoch, avg_cost, linewidth=1.5,
            color=color_cost, label='Cost')
    ax3.fill_between(
        epochs * steps_epoch,
        avg_cost - std_curve_cost,
        avg_cost + std_curve_cost,
        alpha=0.2,
        color=color_cost
    )

    ax3.tick_params(axis='y', labelcolor=color_cost)
    ax3.tick_params(labelsize=16)
    ax3.grid(True)

    fig.suptitle(env, fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])


    if exp == 'auto_update_GA':
        plt.savefig(f"{save_dir}/training_curves_{exp}_{algo}_{env}_{cost_lim=}.pdf")
    if exp == 'auto_update_PID':
        plt.savefig(f"{save_dir}/training_curves_{exp}_{algo}_{env}_{cost_lim=}_{k_p=}_{k_i=}_{k_d=}.pdf")
    
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
    parser.add_argument('--Kp', type=float, default=0.0)
    parser.add_argument('--Ki', type=float, default=1.0)
    parser.add_argument('--Kd', type=float, default=0.0)
    parser.add_argument('--steps_epoch', type=int, default=20000)


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
    k_p = args.Kp
    k_i = args.Ki
    k_d = args.Kd
    steps_epoch = args.steps_epoch



    if exp == 'fixed_lambda':
        if algo != "PPOLag":
            raise ValueError(f"Invalid algo '{algo}'. Expected 'PPOLag'.")
        for lambda_init in lambda_values:
            log_dirs = []
            for seed in seeds:
                log_dir = f'./exp-x/{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}_{lambda_init=}/seed_{seed}'
                log_dirs.append(log_dir)
            plot_training_progress_reward_cost(log_dirs = log_dirs, save_dir = save_dir, env = env, algo = algo, exp=exp, cost_lim=cost_lim, lambda_init = lambda_init, steps_epoch = steps_epoch)


    if exp == 'auto_update_GA':
        if algo != "PPOLag":
            raise ValueError(f"Invalid algo '{algo}'. Expected 'PPOLag'.")
        for seed in seeds:
            log_dir = f'./exp-x/{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}/seed_{seed}'
            log_dirs.append(log_dir)

        plot_training_progress_lambda_reward_cost(log_dirs = log_dirs, save_dir = save_dir, env = env, algo = algo, exp=exp, cost_lim=cost_lim, steps_epoch = steps_epoch)


    if exp == 'auto_update_PID':
        if algo != "CPPOPID":
            raise ValueError(f"Invalid algo '{algo}'. Expected 'CPPOPID'.")
        for seed in seeds:
            log_dir = f'./exp-x/{exp}_{algo}_{env}_{timesteps=}_{cost_lim=}_{k_p=}_{k_i=}_{k_d=}/seed_{seed}'
            log_dirs.append(log_dir)

        plot_training_progress_lambda_reward_cost(log_dirs = log_dirs, save_dir = save_dir, env = env, algo = algo, exp=exp, cost_lim=cost_lim, k_p=k_p, k_i=k_i, k_d=k_d, steps_epoch = steps_epoch)




if __name__ == '__main__':
    main()



