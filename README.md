# High-Confidence Improvement Bayesian Optimization 
This repository contains the code to reproduce the results from the paper **Simulation-Aided Policy Tuning for Black-Box Robot Learning**. 

We propose a general and data-efficient zero-order local search method based on [Gradient Information with Bayesian Optimization (GIBO)](https://github.com/Data-Science-in-Mechanical-Engineering/gibo) with the option to include a low-fidelity information source such as a simulator.

Here is a video of the robot learning experments: [https://www.youtube.com/watch?v=9iSUzwjN7ko](https://www.youtube.com/watch?v=9iSUzwjN7ko).

## Abstract

How can robots learn and adapt to new tasks and situations with little data? Systematic exploration and simulation are crucial tools for efficient robot learning. We present a novel black-box policy search algorithm focused on data-efficient policy improvements. The algorithm learns directly on the robot and treats simulation as an additional information source to speed up the learning process. At the core of the algorithm, a probabilistic model learns the dependence of the policy parameters and the robot learning objective not only by performing experiments on the robot, but also by leveraging data from a simulator. This substantially reduces interaction time with the robot. Using this model, we can guarantee improvements with high probability for each policy update, thereby facilitating fast, goal-oriented learning. We evaluate our algorithm on simulated fine-tuning tasks and demonstrate the data-efficiency of the proposed dual-information source optimization algorithm. In a real robot learning experiment, we show fast and successful task learning on a robot manipulator with the aid of an imperfect simulator. 

## Cite

If you find our code or paper useful, please consider citing the current preprint

```bibtex
@misc{he2024simulation,
  title={Simulation-Aided Policy Tuning for Black-Box Robot Learning}, 
  author={Shiming He and Alexander {von Rohr} and Dominik Baumann and Ji Xiang and Sebastian Trimpe},
  year={2024},
  eprint={2411.14246},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2411.14246}, 
}
```

You may also be interested in the original [GIBO paper](https://proceedings.neurips.cc/paper/2021/hash/ad0f7a25211abc3889cb0f420c85e671-Abstract.html)

```bibtex
@inproceedings{muller2021local,
  author = {M\"{u}ller, Sarah and von Rohr, Alexander and Trimpe, Sebastian},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {20708--20720},
  title = {Local policy search with Bayesian optimization},
  volume = {34},
  year = {2021}
}
```

# Code of the repo
- [optimizers](./src/optimizers.py): Implemented optimizers for black-box functions are [random search](https://arxiv.org/abs/1803.07055), vanilla Bayesian optimization, CMA-ES, GIBO, and the proposed method HCI-GIBO and S-HCI-GIBO.
- [model](./src/model.py): A Gaussian process model with a squared-exponential kernel that also supplies its Jacobian.
- [policy parameterization](./src/policy_parameterizations.py): Multilayer perceptrones as policy parameterization for solving reinforcement learning problems.
- [environment api](./src/environment_api.py): Interface for interactions with reinforcement learning environments of OpenAI Gym.
- [acquisition function](./src/acquisition_function.py): Custom acquisition function for gradient information.
- [loop](./src/loop.py): Brings together all parts necessary for an optimization loop.


# Installation
To run robot experiments, follow the instruction under [franka_pendulum](https://github.com/Data-Science-in-Mechanical-Engineering/franka_pendulum) and [gym-franka-pole](gym-franka-pole/).

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
First launch module `external_acceleration_controller` in [franka_pendulum](https://github.com/Data-Science-in-Mechanical-Engineering/franka_pendulum).

Afterwards you can run
```
python run_robot_experiment.py
```
