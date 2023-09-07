import os
import argparse
import yaml

import numpy as np
import torch

from src import config
from src.loop import loop_IS
from src.synthetic_functions import (
    generate_objective_from_gp_post_IS,
    compute_rewards,
    get_lengthscale_hyperprior,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run optimization of synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")
    parser.add_argument(
        "-cd", "--config_data", type=str, help="Path to data config file."
    )
    parser.add_argument("-d", "--dim", type=int, help="dim")
    parser.add_argument("-i", "--index", type=int, help="index")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Translate config dictionary.
    cfg = config.insert(cfg, config.insertion_config)

    with open(args.config_data, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)

    f_max_dict = torch.load(os.path.join(cfg_data["out_dir"], "f_max.pt"))
    train_x_dict = torch.load(os.path.join(cfg_data["out_dir"], "train_x.pt"))
    train_y_dict = torch.load(os.path.join(cfg_data["out_dir"], "train_y.pt"))
    train_y_IS_dict = torch.load(os.path.join(cfg_data["out_dir"], "train_y_IS1.pt"))
    lengthscales_dict = torch.load(os.path.join(cfg_data["out_dir"], "lengthscales.pt"))
    lengthscales_IS_dict = torch.load(os.path.join(cfg_data["out_dir"], "lengthscales_IS1.pt"))

    params_dict = {}
    calls_dict = {}
    rewards_dict = {}
    points_dict = {}

    #for dim in cfg_data["dimensions"]:
    dim = args.dim
    print(f"\nDimension {dim}.")

    params_list = []
    rewards_list = []
    calls_list = []
    points_list = []

        #for index_objective in range(cfg_data["num_objectives"]):
    index_objective = args.index
    print(f"\nObjective {index_objective+1}.")
    objective = generate_objective_from_gp_post_IS(
                train_x_dict[dim][index_objective],
                train_y_dict[dim][index_objective],
                train_y_IS_dict[dim][index_objective],
                noise_variance=cfg_data["noise_variance"],
                gp_hypers={
                    "covar_module.base_kernel.lengthscale": lengthscales_dict[dim],
                    "covar_module.outputscale": torch.tensor(
                        cfg_data["gp_hypers"]["outputscale"]
                    ),
                },
                gp_hypers_IS={
                    "covar_module.base_kernel.lengthscale": lengthscales_IS_dict[dim],
                    "covar_module.outputscale": torch.tensor(
                        cfg_data["gp_hypers_IS"]["outputscale"]
                    ),
                },
            )
    print(f"Max of objective: {f_max_dict[dim][index_objective]}.")

    hypers = None

    if "set_hypers" in cfg.keys():
        if cfg["set_hypers"]:
            hypers = {
                "covar_module.base_kernel.lengthscale": lengthscales_dict[dim],
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers"]["outputscale"]
               ),
                "covar_module_IS.base_kernel.lengthscale": lengthscales_IS_dict[dim],
                "covar_module_IS.outputscale": torch.tensor(
                    cfg_data["gp_hypers_IS"]["outputscale"]
               ),
                "likelihood.noise": torch.tensor(cfg_data["noise_variance"]),
            }
    elif "only_set_noise_hyper" in cfg.keys():
        if cfg["only_set_noise_hyper"]:
            hypers = {
               "likelihood.noise": torch.tensor(cfg_data["noise_variance"])
            }
    elif "optimize_hyperparameters" in cfg.keys():
        hypers = cfg["optimizer_config"]["hyperparameter_config"]["hypers"]

    cfg_dim = config.evaluate(
        cfg,
        dim_search_space=dim,
        factor_lengthscale=cfg_data["factor_lengthscale"],
        factor_N_max=5,
        hypers=hypers,
    )

    params, calls_in_iteration, sampled_points = loop_IS(
            params_init=torch.cat([0.5 * (torch.ones((1, dim), dtype=torch.float32)), torch.tensor([[0.]])], dim=1), #torch.tensor([0.6, 0.6, 0]), #
                max_iterations=cfg_dim["max_iterations"],
                max_objective_calls=cfg_dim["max_objective_calls"],
                objective=objective,
                Optimizer=cfg_dim["method"],
                optimizer_config=cfg_dim["optimizer_config"],
                verbose=False,
            )

    rewards = compute_rewards(params, objective)
    print(f"Optimizer's max reward: {max(rewards)}")
    params_list.append(params)
    calls_list.append(calls_in_iteration)
    rewards_list.append(rewards)
    points_list.append(sampled_points)

    params_dict[dim] = params_list
    calls_dict[dim] = calls_list
    rewards_dict[dim] = rewards_list
    points_dict[dim] = points_list

    directory = cfg["out_dir"]
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(
        f"Save parameters, objective calls and rewards (function values) at {directory}."
    )
    np.save(os.path.join(directory, "parameters"+'_'+str(dim)+'_'+str(index_objective)), params_dict)
    np.save(os.path.join(directory, "calls"+'_'+str(dim)+'_'+str(index_objective)), calls_dict)
    np.save(os.path.join(directory, "rewards"+'_'+str(dim)+'_'+str(index_objective)), rewards_dict)
    np.save(os.path.join(directory, "sampled_points"+'_'+str(dim)+'_'+str(index_objective)), points_dict)
