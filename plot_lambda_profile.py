

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from typing import List, Optional
import argparse
from statsmodels.nonparametric.smoothers_lowess import lowess



def plot_lambda_profile(log_dirs: Optional[List[str]] = None,
    save_dir: str = '',
    env_id: str = '',
    algo: str = '',
    lambda_values: list = [],
    seeds: list = [],
) -> None:
    
    # retrieving reward and cost training data
    rewards_all_seeds = []
    costs_all_seeds = []

    for seed_num in seeds:

        seed = 'seed_' + str(seed_num)
        all_rewards = []
        all_costs = []

        for log_dir in log_dirs:
            df = pd.read_csv(f"{log_dir}/{seed}/progress.csv")

            all_rewards.append(df['Metrics/EpRet'].iloc[-25:].mean())
            all_costs.append(df['Metrics/EpCost'].iloc[-25:].mean())
        
        rewards_all_seeds.append(all_rewards)
        costs_all_seeds.append(all_costs)

        

    # Average across seeds
    avg_reward = np.mean(np.array(rewards_all_seeds), axis=0)
    avg_cost = np.mean(np.array(costs_all_seeds), axis=0)

    std_curve_reward = np.std(np.array(rewards_all_seeds), axis=0)
    std_curve_cost = np.std(np.array(costs_all_seeds), axis=0)




    # ensure x is sorted
    order = np.argsort(lambda_values)
    x = np.asarray(lambda_values)[order]
    r = np.asarray(avg_reward)[order]
    c = np.asarray(avg_cost)[order]
    r_sd = np.asarray(std_curve_reward)[order]
    c_sd = np.asarray(std_curve_cost)[order]

    # frac is the smoothing strength (span); try between 0.4–0.6
    r_s  = lowess(r,  x, frac=0.4, it=0, return_sorted=False)
    c_s  = lowess(c,  x, frac=0.4, it=0, return_sorted=False)
    r_sd_s = lowess(r_sd, x, frac=0.4, it=0, return_sorted=False)
    c_sd_s = lowess(c_sd, x, frac=0.4, it=0, return_sorted=False)


    # plotting the results
    sns.set(style='white', context='paper', font_scale=1.6) 
    sns.set_palette('deep')  # Optional: color palettes

    # Set up the figure and first axis
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot reward on the left Y-axis
    color_reward = 'mediumblue'
    ax1.set_xlabel(r'$\lambda$', fontsize=18)
    ax1.set_ylabel('Reward', fontsize=18, color=color_reward)
    ax1.plot(x, r_s, linewidth=1.5, color=color_reward, label='Reward')
    ax1.fill_between(x, r_s - r_sd_s, r_s + r_sd_s, alpha=0.1, color=color_reward)
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    # Plot cost on the right Y-axis
    color_cost = 'crimson'
    ax2.set_ylabel('Cost', fontsize=18, color=color_cost)
    ax2.plot(x, c_s, linewidth=1.5, color=color_cost, label='Cost')
    ax2.fill_between(x, c_s - c_sd_s, c_s + c_sd_s, alpha=0.1, color=color_cost)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.set_ylim(bottom=0)

    plt.title(env_id, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax1.grid(True)
    fig.tight_layout()

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

    for lambda_init in lambda_values:
        log_dir = f'./exp-x/fixed_lambda_{algo}_{env}_{timesteps=}_{cost_lim=}_{lambda_init=}'
        log_dirs.append(log_dir)

    plot_lambda_profile(log_dirs = log_dirs, save_dir = save_dir, env_id = env, algo = algo, lambda_values = lambda_values, seeds = seeds)

    

if __name__ == '__main__':
    main()



