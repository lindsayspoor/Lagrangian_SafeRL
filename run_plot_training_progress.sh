#!/bin/bash

# Set your base directory
BASE_DIR="exp-x"

# Extract all unique lambda values from folder names
LAMBDAS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "*lambda_init*" \
    | sed -E 's|.*lambda_init=([0-9.]+).*|\1|' \
    | sort -u)

# Extract all unique seeds from folder names (! assumes that each folder contains the same amount of seeds !)
SEEDS=$(find "$BASE_DIR" -type d -name "seed_*" \
    | sed -E 's|.*seed_([0-9]+).*|\1|' \
    | sort -nu)


# Specify the experiment settings you want to produce training curves of
# (should be the same as the settings you specified to train your models on)

python plot_training_progress.py \
    --exp "auto_update_PID_rsi" \
    --lambda_values $LAMBDAS \
    --algo "CPPOPIDRSI" \
    --cost_lim 25.0 \
    --env "SafetyAntVelocity-v1" \
    --save_dir "./" \
    --timesteps 35000000 \
    --seeds $SEEDS \
    --Kp 0.0001 \
    --Ki 0.0001 \
    --Kd 0.0 \
    --lambda_init 1.0 \




echo "Done."