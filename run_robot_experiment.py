from re import T
from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

import os
import sys
from datetime import date

import numpy as np
import scipy

import torch
import botorch
import gpytorch

import gym
import gym_franka_pole

# To import module code.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.environment_api import EnvironmentObjective, StateNormalizer
from src.policy_parameterizations import MLP, DMP, MIX, discretize
from src.loop import loop
from src.optimizers import BayesianGradientAscent, TrustBayesianGradientAscent, MIBayesianGradientAscent
from src.model import DerivativeExactGPSEModel, MIDerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_custom_bo, optimize_acqf_MI_bo

import matplotlib.pyplot as plt

env = gym.make("franka_pole-v0")

dmp = DMP(2, 12, time=20.)

len_params = dmp.len_params

objective_env = EnvironmentObjective(env=env,
                                     policy=dmp,
                                     manipulate_state=None,
                                     manipulate_reward=lambda r, a, s, d: r / 100.,
                                     )

parameters = {}
calls = {}

for trial in range(1):
    print(f'trial {trial+1}')
    params, calls_in_iteration = loop(params_init=torch.zeros(1, len_params+1),
                                      max_iterations=None,
                                      max_objective_calls=1000,
                                      objective=objective_env,
                                      Optimizer=MIBayesianGradientAscent,
                                      optimizer_config={'max_samples_per_iteration': len_params,
                                                        'OptimizerTorch': torch.optim.SGD,
                                                        'optimizer_torch_config': {'lr': 1.0},
                                                        'lr_schedular': {0: 0.5, 4: 0.25, 7: 0.1},
                                                        'alpha': 0.95,
                                                        'Model': MIDerivativeExactGPSEModel,
                                                        'model_config': {'lengthscale_constraint':None,
                                                        'lengthscale_hyperprior': None, 
                                                        'lengthscale_constraint': gpytorch.constraints.Interval(0.1, 0.3),
                                                        'lengthscale_hyperprior_IS': None,
                                                        'lengthscale_constraint_IS': gpytorch.constraints.Interval(0.05, 0.3),
                                                        'outputscale_constraint': gpytorch.constraints.Interval(1., 4.),
                                                        'outputscale_hyperprior': None, 
                                                        'outputscale_hyperprior_IS': None, 
                                                        'outputscale_constraint_IS': gpytorch.constraints.Interval(0.2, 0.8),
                                                        'noise_constraint': None, 
                                                        'noise_hyperprior': None,
                                                        'N_max': 40,
                                                        'ard_num_dims':len_params,
                                                        'prior_mean':0}, 
                                                        'hyperparameter_config': {'optimize_hyperparameters': False,
                                                                                  'hypers': {'covar_module.base_kernel.lengthscale': 0.2*torch.ones(len_params),
                                                                                             'covar_module_IS.base_kernel.lengthscale': 0.3*torch.ones(len_params),
                                                                                             'covar_module.outputscale': torch.tensor(3.),
                                                                                             'covar_module_IS.outputscale': torch.tensor(0.5),
                                                                                             'likelihood.noise': torch.tensor(0.01)},
                                                                                  'no_noise_optimization': True},
                                                        'optimize_acqf': optimize_acqf_MI_bo,
                                                        'optimize_acqf_config': {'q': 1,
                                                                                 'num_restarts': 5,
                                                                                 'raw_samples': 64},
                                                        'bounds': None,
                                                        'delta': 0.15,
                                                        'epsilon_diff_acq_value': 24,
                                                        'generate_initial_data': None,
                                                        'standard_deviation_scaling': False,
                                                        'normalize_gradient': True,
                                                        'verbose': True,
                                                       },
                                      verbose=True)
    parameters[trial] = torch.cat(params)
    calls[trial] = calls_in_iteration

