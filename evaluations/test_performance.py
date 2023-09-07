from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List
import os
import sys
import yaml

import copy

import numpy as np
import torch
from torch import Tensor

# To import module code.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.evaluate import (
    sort_rewards_global_optimization,
    postprocessing_interpolation_rewards,
    f_max_new
)

import matplotlib.pyplot as plt


def postprocess_data(configs: List[str],
                     experiment_path: int,
                     sort_rewards: bool = True,
                     interpolate: bool = True,
                     max_new: bool = False,
                     ):
    method_to_name = {'vbo': 'Vanilla BO', 'rs': 'ARS', 'crbo': 'CRBO', 'hci-gibo': 'HCI-GIBO', 'gibo': 'GIBO', 's-hci-gibo': 'S-HCI-GIBO'}
    list_interpolated_rewards = []
    list_names_optimizer = []

    for cfg_str in configs:

        with open(cfg_str, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        directory = '.' + cfg['out_dir']
        points = []
        calls = []
        if interpolate:

            print('Postprocess tracked parameters over optimization procedure.')

            # Load data.
            print(f'Load data from {directory}.')
            parameters = np.load(
                os.path.join(directory, 'parameters.npy'), allow_pickle=True
            ).item()
            rewards = np.load(
                os.path.join(directory, 'rewards.npy'), allow_pickle=True
            ).item()
            calls = np.load(
                os.path.join(directory, 'calls.npy'), allow_pickle=True
            ).item()
            if sort_rewards:
                rewards = sort_rewards_global_optimization(rewards)

            # Postprocess data (offline evaluation and interpolation).
            print('Postprocess data: interpolate.')
            interpolated_rewards = postprocessing_interpolation_rewards(
                rewards, calls, calls_of_objective=cfg['max_objective_calls']
            )

            # Save postprocessed data.
            print(f'Save postprocessed data in {directory}.')
            torch.save(
                interpolated_rewards, os.path.join(directory, 'interpolated_rewards.pt')
            )

        else:
            interpolated_rewards = torch.load(
                os.path.join(directory, 'interpolated_rewards.pt')
            )

        list_names_optimizer.append(method_to_name[cfg['method']])
        list_interpolated_rewards.append(interpolated_rewards)

    f_max_dict = torch.load(experiment_path + '/f_max.pt')
    if max_new:
        f_max_new(f_max_dict, list_interpolated_rewards)

    return list_names_optimizer, list_interpolated_rewards, f_max_dict, points, calls

(list_names_optimizer,
 list_interpolated_rewards,
 f_max_dict,
 points,
 calls) = postprocess_data(configs=[#'../configs/synthetic_experiment/vbo_default.yaml',
                                    #'../configs/synthetic_experiment/rs_default.yaml',
                                    #'../configs/synthetic_experiment/crbo_default.yaml',
                                    '../configs/synthetic_experiment/gibo_default.yaml',
                                    '../configs/synthetic_experiment/hci-gibo_default.yaml',
                                    #'../configs/synthetic_experiment/s-hci-gibo_default.yaml',
                                    ],
                                    experiment_path='../experiments/synthetic_experiments/test/',
                                    sort_rewards=True,
                                    interpolate=True)


def plot_function_value_distances(
        f_max: Dict,
        rewards_optimizers: List[Tensor],
        names_optimizers: List[str],
        figsize: Tuple[Union[int, float]],
        row_col: Tuple[int] = (2, 3),
        path_savefig: Optional[str] = None,
        remove_dim_index: Optional[List[int]] = None,
):
    markers = ["D", "o", "s", ">", "*", "+", "H"]
    dimensions = list(f_max.keys())
    dim_indices = torch.arange(0, len(dimensions))


    num_objective_calls = rewards_optimizers[0].shape[-1]
    num_optimizers = len(rewards_optimizers)
    n_rows, n_cols = row_col
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=figsize)
    fig.subplots_adjust(hspace=0.3)
    axs = np.array(axs).reshape(-1)
    for index_dim, dim in enumerate(dimensions):
        f_max_dim = f_max[dimensions[index_dim]]
        f_max_reshaped = torch.reshape(torch.tensor(f_max_dim), (-1, 1))
        #axs[index_dim].set_yscale("log")
        axs[index_dim].set_title(f"{dim}-dim. domain", fontsize=10)
        axs[index_dim].set_xlim([0, num_objective_calls])
        axs[index_dim].set_ylim([0, 1.5])
        axs[index_dim].yaxis.set_ticks(np.arange(0, 0.95, 0.2))
        #axs[index_dim].yaxis.set_major_locator(ticker.MultipleLocator(1))
        for index_optimizer, rewards in enumerate(rewards_optimizers):
            value_error = f_max_reshaped - rewards[dim_indices, :, :][index_dim]
            value_error_normalized = 1 - value_error / f_max_reshaped
            mean_x = torch.zeros(101)
            std_x = torch.zeros(101)
            for i in range(101):
                reshaped_value = torch.sum(value_error_normalized < (i / 100.), dim=1).type(torch.DoubleTensor)
                mean_x[i] = torch.mean(reshaped_value)
                std_x[i] = torch.std(reshaped_value)

            axs[index_dim].plot(
                mean_x,
                torch.linspace(0, 1, 101),
                label=names_optimizers[index_optimizer],
                marker=markers[index_optimizer],
                linestyle="-",
                linewidth=1.,
                markersize=3,
                markeredgewidth=0.6,
                markevery=25,
                fillstyle="none",
            )
            axs[index_dim].fill_betweenx(
                torch.linspace(0, 1, 101),
                mean_x - std_x,
                mean_x + std_x,
                alpha=0.2,
            )
            axs[index_dim].set_xlim([0, 200])
            axs[index_dim].set_ylim([0, 0.99])
            axs[index_dim].xaxis.grid(True, linewidth=.4)
            axs[index_dim].yaxis.grid(True, linewidth=.4)
    handles, labels = axs[index_dim].get_legend_handles_labels()
    axs[0].set_ylabel(r"$\mathcal{SA}$")
    axs[3].set_ylabel(r"$\mathcal{SA}$")
    axs[3].set_xlabel("\# of evaluations")
    axs[4].set_xlabel("\# of evaluations")
    axs[5].set_xlabel("\# of evaluations")
    lgd = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=num_optimizers,
        frameon=False,
    )
    if path_savefig:
        plt.savefig(path_savefig, bbox_inches="tight", bbox_extra_artists=[lgd])
    plt.show()

plt.style.use('seaborn-whitegrid')
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}


plt.rcParams.update(tex_fonts)
#%%
plot_function_value_distances(f_max=f_max_dict,
                              rewards_optimizers=list_interpolated_rewards,
                              names_optimizers=list_names_optimizer,
                              figsize=(12, 3),
                              path_savefig='../experiments/synthetic_experiments/test/within-model.pdf',
                              )

