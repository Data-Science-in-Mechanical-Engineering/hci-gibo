import os
import yaml
import argparse

import torch

from src.synthetic_functions import (
    generate_training_samples_IS,
    get_maxima_objectives,
    get_lengthscales,
    factor_hennig,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate data for synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")


    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg_data = yaml.load(f, Loader=yaml.Loader)

    train_x_dict = {}
    train_y_dict = {}
    train_y_IS_dict = {}
    train_y_IS1_dict = {}
    train_y_IS2_dict = {}
    train_y_IS3_dict = {}
    lengthscales_dict = {}
    lengthscales_IS_dict = {}
    lengthscales_IS1_dict = {}
    lengthscales_IS2_dict = {}
    lengthscales_IS3_dict = {}

    print(
        "Generate data (train_x, train_y, lengthscales, f_max, argmax) for synthetic test functions with domains of different dimensionality."
    )

    for dim in cfg_data["dimensions"]:
        print(f"Data for function with {dim}-dimensional domain.")
        l = get_lengthscales(dim, factor_hennig)
        m = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]),
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]),
        )
        m_IS = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]) * 0.5,
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]) * 0.5,
        )
        m_IS1 = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]) * 1,
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]) * 1,
        )
        m_IS2 = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]) * 2,
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]) * 2,
        )
        m_IS3 = torch.distributions.Uniform(
            cfg_data["factor_lengthscale"] * l * (1 - cfg_data["gamma"]) * 4,
            cfg_data["factor_lengthscale"] * l * (1 + cfg_data["gamma"]) * 4,
        )
        lengthscale = m.sample((1, dim))
        lengthscale_IS = m_IS.sample((1, dim))
        lengthscale_IS1 = m_IS1.sample((1, dim))
        lengthscale_IS2 = m_IS2.sample((1, dim))
        lengthscale_IS3 = m_IS3.sample((1, dim))
        train_x, train_y, train_y_IS, train_y_IS1, train_y_IS2, train_y_IS3 = generate_training_samples_IS(
            num_objectives=cfg_data["num_objectives"],
            dim=dim,
            num_samples=cfg_data["num_samples"],
            gp_hypers={
                "covar_module.base_kernel.lengthscale": lengthscale,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers"]["outputscale"]
                ),
            },
            gp_hypers_IS={
                "covar_module.base_kernel.lengthscale": lengthscale_IS,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers_IS"]["outputscale"]
                ),
            },
            gp_hypers_IS1={
                "covar_module.base_kernel.lengthscale": lengthscale_IS1,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers_IS"]["outputscale"]
                ),
            },
            gp_hypers_IS2={
                "covar_module.base_kernel.lengthscale": lengthscale_IS2,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers_IS"]["outputscale"]
                ),
            },
            gp_hypers_IS3={
                "covar_module.base_kernel.lengthscale": lengthscale_IS3,
                "covar_module.outputscale": torch.tensor(
                    cfg_data["gp_hypers_IS"]["outputscale"]
                ),
            },
        )
        train_x_dict[dim] = train_x
        train_y_dict[dim] = train_y
        train_y_IS_dict[dim] = train_y_IS
        train_y_IS1_dict[dim] = train_y_IS1
        train_y_IS2_dict[dim] = train_y_IS2
        train_y_IS3_dict[dim] = train_y_IS3
        lengthscales_dict[dim] = lengthscale
        lengthscales_IS_dict[dim] = lengthscale_IS
        lengthscales_IS1_dict[dim] = lengthscale_IS1
        lengthscales_IS2_dict[dim] = lengthscale_IS2
        lengthscales_IS3_dict[dim] = lengthscale_IS3


    print("Compute maxima and argmax of synthetic functions.")
    f_max_dict, argmax_dict = get_maxima_objectives(
        lengthscales=lengthscales_dict,
        noise_variance=cfg_data["noise_variance"],
        train_x=train_x_dict,
        train_y=train_y_dict,
        n_max=cfg_data["n_max"],
    )

    if not os.path.exists(cfg_data["out_dir"]):
        os.mkdir(cfg_data["out_dir"])

    path = cfg_data["out_dir"]
    print(f"Save data at {path}.")
    torch.save(train_x_dict, os.path.join(path, "train_x.pt"))
    torch.save(train_y_dict, os.path.join(path, "train_y.pt"))
    torch.save(train_y_IS_dict, os.path.join(path, "train_y_IS.pt"))
    torch.save(train_y_IS1_dict, os.path.join(path, "train_y_IS1.pt"))
    torch.save(train_y_IS2_dict, os.path.join(path, "train_y_IS2.pt"))
    torch.save(train_y_IS3_dict, os.path.join(path, "train_y_IS3.pt"))
    torch.save(lengthscales_dict, os.path.join(path, "lengthscales.pt"))
    torch.save(lengthscales_IS_dict, os.path.join(path, "lengthscales_IS.pt"))
    torch.save(lengthscales_IS1_dict, os.path.join(path, "lengthscales_IS1.pt"))
    torch.save(lengthscales_IS2_dict, os.path.join(path, "lengthscales_IS2.pt"))
    torch.save(lengthscales_IS3_dict, os.path.join(path, "lengthscales_IS3.pt"))
    torch.save(f_max_dict, os.path.join(path, "f_max.pt"))
    torch.save(argmax_dict, os.path.join(path, "argmax.pt"))

