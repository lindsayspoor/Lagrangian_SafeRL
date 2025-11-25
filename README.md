

# Lagrangian Safe RL

  
This repository provides codebase for our project on **Lagrangian-based Safe Reinforcement Learning (Safe RL)**.

It is intended for **reproducing the results** from our paper and for getting started with your own empirical analysis of Lagrangian multiplier behavior in Safe RL.

**Paper:** *An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning*
**Authors:** L. Spoor, Á. Serra-Gómez, A. Plaat, T. Moerland
**arXiv:** https://arxiv.org/pdf/2510.17564
**Corresponding author:** l.j.spoor@liacs.leidenuniv.nl

---
  
## Dependencies

This project relies heavily on two benchmark suites, Omnisafe and Safety-Gymnasium:

If you run into benchmark-specific issues, please consult their documentation

- https://omnisafe.readthedocs.io/en/latest/index.html
- https://safety-gymnasium.readthedocs.io/en/latest/

---
## Installation

```bash

conda create -n omnisafe python=3.8

conda activate omnisafe

pip install omnisafe

````
  
---
## Make scripts executable

Before running experiments or plots, ensure all shell scripts are executable:

```bash

chmod -x run_plot_lambda_profile.sh

chmod -x run_plot_training_progress.sh

  

cd exp-x

chmod -x collect_all_seeds.sh

cd ..

```
---
## Running Experiments

Experiments are launched via:

```bash

python run_test_experiments.py

```  

You can specify settings through command-line arguments.


### Example: fixed λ experiment

To train models with:

* fixed multiplier λ = 0.03
* cost limit = 25.0
* environment = `SafetyPointCircle1-v0`
* seed = 0
* timesteps = 100000

run:

```bash

python run_test_experiments.py \

--exp "fixed_lambda" \

--algo "PPOLag" \

--env "SafetyPointCircle1-v0" \

--cost_lim 25.0 \

--lambda_init 0.03 \

--seed 0 \

--timesteps 100000

```

### Automated multiplier updates

* **Gradient ascent updates** (PPOLag):

```bash

--exp "auto_update_GA" --algo "PPOLag"

```

* **PID-controlled updates** (CPPOPID):

```bash

--exp "auto_update_PID" --algo "CPPOPID" --Kp 0.0001 --Ki 0.0001 --Kd 0.0

```

---

## Collecting results

All training output is stored under `exp-x/`.
After you have trained your models, extract and aggregate per-seed training logs:


```bash

cd exp-x

./collect_all_seeds.sh

cd ..

```

  
This parses each seed’s `progress.csv` and prepares the files for downstream analysis.

  

---

  

## Plotting λ-profiles

To generate λ-profile plots over fixed multiplier runs:


```bash

./run_plot_lambda_profile.sh

```

  

Inside that script, set arguments matching the training configuration.
For the example above:


```bash

python plot_lambda_profile.py \

--lambda_values $LAMBDAS \

--algo "PPOLag" \

--cost_lim 25.0 \

--env "SafetyPointCircle1-v0" \

--save_dir "./plots/lambda_profiles/" \

--timesteps 100000 \

--seeds $SEEDS

```

**Important note.**

This script assumes **each fixed-λ folder contains the same set of seeds**.

It finds all unique λ folders, sweeps over all seeds within each folder, and constructs a λ-profile by averaging across seeds.

---

## Plotting training curves

To plot training curves (reward/cost over time):

```bash

./run_plot_training_progress.sh

```

Edit the script to match the experiment you want to visualize.

---

  

## Command-line arguments

  
### `run_test_experiments.py`


* `--seed` *(int, default=0)*: Seed for an individual run.
* `--env` *(str, default="SafetyPointCircle1-v0")*: Any Safety Gymnasium MuJoCo environment.
* `--algo` *(str, default="PPOLag")*: `"PPOLag"`: fixed multiplier or GA-updated multiplier; `"CPPOPID"`: PID-controlled multiplier updates
* `--lambda_init` *(float, default=0.03)*: Initial multiplier value. For fixed-λ experiments, this is the constant value used throughout training.
* `--exp` *(str, default="fixed_lambda")*: Experiment type: `fixed_lambda`: multiplier fixed during training; `auto_update_GA`: gradient-ascent multiplier updates; `auto_update_PID`: PID-controlled multiplier updates
* `--Kp` *(float, default=0.0)*: Proportional gain (only for `auto_update_PID`)
* `--Ki` *(float, default=1.0)*: Integral gain (only for `auto_update_PID`).
* `--Kd` *(float, default=0.0)*: Derivative gain (only for `auto_update_PID`).
* `--cost_lim` *(float, default=25.0)*:  Cost limit the algorithm targets (relevant for GA/PID updates).
* `--timesteps` *(int, default=100000)*: Total training timesteps.

* `--venvs` *(int, default=5)*: Number of vectorized environments.
* `--torchthr` *(int, default=1)*: Number of PyTorch threads.
* `--numpool` *(int, default=1)*: Multiprocessing pool size.
* `--steps_epoch` *(int, default=20000)*:Training steps per epoch.

---

### `run_plot_lambda_profile.sh` / `plot_lambda_profile.py`

* `--lambda_values` *(default: `$LAMBDAS`)*: List like `0.03 0.05 0.07`, or use the auto-extracted defaults
* `--algo` *(default: "PPOLag")*: Only supports `"PPOLag"`.
* `--cost_lim` *(default: 25.0)*: Must match the trained models.
* `--env` *(default: "SafetyPointCircle1-v0")*: Must match the trained models.
* `--save_dir` *(default: "./plots/lambda_profiles/")*: Output directory.
* `--timesteps` *(default: 40000)*: Total training timesteps used by the models.
* `--seeds` *(default: `$SEEDS`)*: Seeds to average over (e.g., `0 1 2 4 6`).

---

  

### `run_plot_training_progress.sh`

* `--exp` *(default: "auto_update_PID")* : Which experiment to plot: `fixed_lambda`, `auto_update_GA`, `auto_update_PID

* `--lambda_values` *(default: `$LAMBDAS`)*: Only relevant for `fixed_lambda`; list which λ values to plot
* `--algo` *(default: "CPPOPID")*: `"PPOLag"` for `fixed_lambda` and `auto_update_GA`; `"CPPOPID"` for `auto_update_PID`
* `--cost_lim` *(default: 25.0)*: Must match the trained models.
* `--env` *(default: "SafetyPointCircle1-v0")*: Must match the trained models.
* `--save_dir` *(default: "./plots/training_curves/")*: Output directory.
* `--timesteps` *(default: 40000)*: Timesteps used during training.
* `--seeds` *(default: `$SEEDS`)*: Seeds to include (e.g., `1 5 8`).
* `--Kp` *(default: 0.0001)*: Only for `auto_update_PID`; must match training.
* `--Ki` *(default: 0.0001)*: Only for `auto_update_PID`; must match training.
* `--Kd` *(default: 0.0)*: Only for `auto_update_PID`; must match training.

---

## Citation

If you use this repository in academic work, please cite:

Spoor, L., Serra-Gómez, Á., Plaat, A., & Moerland, T.
*An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning.* arXiv:2510.17564, 2025.

---

  

## Contact

For questions, issues, or collaboration, please contact:

**Lindsay Spoor**
l.j.spoor@liacs.leidenuniv.nl
