# High-Confidence Improvement Bayesion Optimization 
This repository contains the code to reproduce the results from the paper **Towards Data-Efficient Robot Learning
with the Help of Simulators**.

We propose a general and data-efficient zero-order local search method based on [GIBO](https://github.com/sarmueller/gibo/).

# Code of the repo
- [optimizers](./src/optimizers.py): Implemented optimizers for black-box functions are [random search](https://arxiv.org/abs/1803.07055), vanilla Bayesian optimization, CMA-ES, GIBO, and the proposed method HCI-GIBO and S-HCI-GIBO.
- [model](./src/model.py): A Gaussian process model with a squared-exponential kernel that also supplies its Jacobian.
- [policy parameterization](./src/policy_parameterizations.py): Multilayer perceptrones as policy parameterization for solving reinforcement learning problems.
- [environment api](./src/environment_api.py): Interface for interactions with reinforcement learning environments of OpenAI Gym.
- [acquisition function](./src/acquisition_function.py): Custom acquisition function for gradient information.
- [loop](./src/loop.py): Brings together all parts necessary for an optimization loop.


# Installation
To run robot experiments, follow the instruction under [franka_pendulum](https://github.com/data-Science-in-Mechanical-Engineering/franka_pendulum) and [gym-franka-pole](gym-franka-pole/).

Into an environment with python 3.7.3 you can install all needed packages with
```
pip install -r requirements.txt
```

# Usage 
For experiments with synthetic test functions with dual information sources. a command-line interface is supplied.

## Synthetic Test Functions
### Run
First generate the needed data for the synthetic test functions.

```
python generate_data_synthetic_functions_IS.py -c ./configs/synthetic_experiment/generate_data_MIS.yaml
```

Afterwards you can run for instance GIBO, HCI-GIBO and S-HCI-GIBO on these test functions.

```
# GIBO
python run_synthetic_experiment.py -c ./configs/synthetic_experiment/gibo_default.yaml -cd ./configs/synthetic_experiment/generate_data_MIS.yaml
```
```
# HCI-GIBO
python run_synthetic_experiment.py -c ./configs/synthetic_experiment/hci-gibo_default.yaml -cd ./configs/synthetic_experiment/generate_data_MIS.yaml
```
```
# S-HCI-GIBO
python run_synthetic_experiment_IS.py -c ./configs/synthetic_experiment/s-hci-gibo_default.yaml -cd ./configs/synthetic_experiment/generate_data_MIS.yaml
```
### Evaluate
Evaluation of the synthetic experiments:
```
python evaluations/test_performance.py
```

## Robot Experiment
### Run
First launch module `external_acceleration_controller` in [franka_pendulum](https://github.com/data-Science-in-Mechanical-Engineering/franka_pendulum).

Afterwards you can run
```
python run_robot_experiment.py
```