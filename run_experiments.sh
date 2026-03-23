#!/bin/bash

# Rollout parallelism: match venvs to CPUs reasonably




# '--seed', type=int, default=0

# '--env', type=str, default='SafetyPointCircle1-v0', choice of all env id's supported by Safety Gymnasium's task suite

# '--algo', type=str, default='PPOLag', choice of: no reward-scale-invariance: 'PPOLag', 'SACLag', 'TRPOLag', 'DDPGLag', 'TD3Lag', 'CPPOPID', 'TRPOPID', 
# 'DDPGPID', 'SACPID', 'TD3PID';
# with reward-scale-invariance (currently only implemented for PPO-Lag, SAC-Lag and CPPOPID-Lag): 'PPOLagRSI', 'SACLagRSI', 'CPPOPIDRSI'

# --exp: experiment setting to use, choice of: no reward-scale-invariance: 'fixed_lamnda', 'auto_update_GA', 'auto_update_PID'; 
# with reward-scale-invariance(currently only implemented for PPO-Lag, SAC-Lag and CPPOPID-Lag): 'fixed_lambda_rsi', 'fixed_lambda_rsi_sac' (SAC-Lag specific), 
# 'auto_update_GA_rsi', 'auto_update_GA_rsi_sac' (SAC-Lag specific), 'auto_update_PID_rsi' 

# '--lambda_init', type=float, default=1.0


# '--Kp', type=float, default=0.0, proportional gain term of the PID update mechanism
# '--Ki', type=float, default=1.0, integral gain term of the PID update mechanism
# '--Kd', type=float, default=0.0, derivative gain term of the PID update mechanism

# '--cost_lim', type=float, default=25.0, cost limit the Lagrangian-based algorithm will approach

# '--timesteps', type=int, default=10000000
# '--slurm_job_id', type=int, default=0
# '--venvs', type=int, default=10
# '--torchthr', type=int, default=1
# '--numpool', type=int, default=1
# '--steps_epoch', type=int, default=20000
# '--hidden_sizes', type=int, nargs='+', default = [512,512]
# '--activation', type=str, default ='elu'


python experiments.py \
    --seed 0 \
    --env "SafetyPointPush1-v0" \
    --exp "fixed_lambda_rsi" \
    --cost_lim 25.0 \
    --venvs 16 \
    --algo "PPOLagRSI" \
    --lambda_init 0.1 \
    --timesteps 35000000 \
    --torchthr 2

echo "Done."