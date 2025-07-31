import os
import sys
from datetime import datetime
import argparse

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple



def train(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    # pylint: disable-next=consider-using-with
    sys.stdout = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', terminal_log_name),
        'w',
        encoding='utf-8',
    )
    # pylint: disable-next=consider-using-with
    sys.stderr = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', error_log_name),
        'w',
        encoding='utf-8',
    )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='SafetyPointCircle1-v0')
    parser.add_argument('--algo', type=str, default='PPOLag')
    parser.add_argument('--venvs', type=int, default=2)
    parser.add_argument('--torchthr', type=int, default=1)
    parser.add_argument('--numpool', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=10000000)
    parser.add_argument('--steps_epoch', type=int, default=20000)
    parser.add_argument('--lambda_init', type=float, default=0.03)
    parser.add_argument('--exp', type=str, default='lambda_range')

    return parser.parse_args()

def main():


    args = parse_args()
    seed = args.seed
    algo = args.algo
    env = args.env
    venvs = args.venvs
    torchthr = args.torchthr
    numpool = args.numpool
    timesteps = args.timesteps
    steps_epoch = args.steps_epoch
    lambda_init = args.lambda_init
    exp = args.exp

    if exp == 'fixed_lambda':
        eg = ExperimentGrid(exp_name=f'{algo}_{env}_{exp}_{lambda_init}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/seed_{seed}/')
        eg.add('lagrange_cfgs:lagrangian_multiplier_init', [lambda_init])
        eg.add('lagrange_cfgs:lambda_lr', [0.0])
    
    else:
        eg = ExperimentGrid(exp_name=f'{algo}_{env}_{exp}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}/seed_{seed}/')

    # Set the algorithms.
    base_policy = [algo]

    # Set the environments.
    mujoco_envs = [env]#, 'SafetyPointGoal1-v0'] #, 'SafetyPointButton1-v0', 'SafetyPointPush1-v0'] # Safe Navigation level 1 all tasks with Point agent
    eg.add('env_id', mujoco_envs)
    eg.add('algo', base_policy)
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('train_cfgs:vector_env_nums', [venvs])
    eg.add('train_cfgs:torch_threads', [torchthr])
    eg.add('train_cfgs:total_steps', [timesteps])
    eg.add('algo_cfgs:steps_per_epoch', [steps_epoch])
    eg.add('seed', [seed])

    eg.run(train, num_pool=numpool)

    eg.analyze(parameter='algo', values=None, compare_num=None, cost_limit=None)



if __name__ == '__main__':
    main()