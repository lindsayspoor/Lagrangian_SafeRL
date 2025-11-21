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


# Specify the experiment settings you want to evaluate and make plots on 
# (should be the same as the settings you specified to train your models on)
python plot_lambda_profile.py \
        --lambda_values $LAMBDAS \
        --algo "PPOLag" \
        --cost_lim 25.0 \
        --env "SafetyPointCircle1-v0" \
        --save_dir "./plots/lambda_profiles/" \
        --timesteps 40000 \
        --seeds $SEEDS

echo "Done."