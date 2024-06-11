# averages (algined) models and evaluates their performance

import torch
import copy
from typing import List
from collections import OrderedDict
from ptmz.model_averaging.align_models import align_models
from ptmz.model_averaging.checkpoint_averaging import average_model_checkpoints
from ptmz.model_averagingevaluate_single_model import evaluate_single_model
from ptmz.model_averagingcondition_bn import condition_bn
from ptmz.model_averagingload_dataset import load_datasets_from_config

from pathlib import Path


import numpy as np
import json


def rename_state_keys(state, key_old, key_new):
    new_state = OrderedDict()

    for key in state.keys():
        nkey = key.replace(key_old, key_new)
        new_state[nkey] = state[key]
    return new_state

def evaluate_model_interpolation(
    config: dict,
    device: torch.device = None,
    verbosity: int = 15,
) -> dict:
    """
    Computes the linear model combination of two models and evaluates the performance.
    Args:
        config (dict): config dictionary. Must contain the following
            - model_paths (list): list of paths to model configs
            - epoch: epoch of checkpoint to load
            - align_models (bool): flag to align models
            - interpolation_steps (int): number of interpolation steps
        device (torch.device): device to use for computation
        verbosity (int): verbosity level
    Returns:
        Evaluation results
    """
    interpolation_steps = config.get("interpolation_steps", 100)
    align_models_flag = config.get("align_models", True)
    finetuning_epochs = config.get("finetuning_epochs", 0)

    # finetuning_epochs: int = 0,

    # configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda" or device == torch.device("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print(f"Warning: device {device} is not available. Using CPU instead.")
            device = torch.device("cpu")

    # extract checkopint eopch
    epoch = config["epoch"]

    # extract model config
    config_paths = [
        Path(mpdx).joinpath("params.json") for mpdx in config["model_paths"]
    ]
    configs = [json.load(Path(config_pdx).open("r")) for config_pdx in config_paths]
    for config_dx in configs:
        config_dx["device"] = device
        config_dx["cuda"] = device == torch.device("cuda")
        config_dx["scheduler::steps_per_epoch"] = 123

    # load data
    trainloader, _, _ = load_datasets_from_config(configs[0])

    # load checkpoints
    checkpoint_paths = [
        Path(mpdx).joinpath(f"checkpoint_{epoch:06d}", "checkpoints")
        for mpdx in config["model_paths"]
    ]

    # load checkpoints
    print("load checkpoints")
    load_device = "cpu"
    checkpoints = [
        torch.load(path, map_location=load_device) for path in checkpoint_paths
    ]
    # rename checkpoint keys
    checkpoints = [rename_state_keys(check, "module.", "") for check in checkpoints]

    # align models
    if align_models_flag:
        print(f"align models")
        checkpoints = align_models(configs[0], checkpoints)

    results = {
        "individual": [],
    }


    # iterate over unique model pairs
    for idx, checkpoint_a in enumerate(checkpoints):
        for jdx, checkpoint_b in enumerate(checkpoints[idx + 1 :]):
            # compute linear model combination
            res_tmp = interpolate_model_pair(
                checkpoint_a=checkpoint_a,
                checkpoint_b=checkpoint_b,
                dataloader=trainloader,
                interpolation_steps=interpolation_steps,
            )
            res_tmp["model_a"] = config["model_paths"][idx]
            res_tmp["model_b"] = config["model_paths"][jdx]
            results["individual"].append(res_tmp)

    # aggregate numerical values in dict of lists in average
    results["aggregated"] = {}
    results["aggregated"]["alpha"] = results["individual"][0]["alpha"]
    # iterate over output keys
    for key in ['loss_train', 'loss_val', 'loss_test', 'acc_train', 'acc_val', 'acc_test']:
        if key in results['individual'][0]:
            results["aggregated"][key] = []
            for idx in range(len(results["individual"])):
                results["aggregated"][key].append(results["individual"][idx][key])
            # compute mean per entry so that we have a list of mean entries
            results["aggregated"][key] = np.mean(results["aggregated"][key], axis=0)

    return results

def interpolate_model_pair(
    checkpoint_a, checkpoint_b, config, interpolation_steps, dataloader
) -> dict:
""" Computes the linear model combination of two models and evaluates the performance.
"""

    # interpolate models
    for alpha in np.linspace(0, 1, interpolation_steps):
        averaged_model = average_model_checkpoints(
            [checkpoint_a, checkpoint_b], [alpha, 1 - alpha]
        )
        # apply batch norm conditioning
        print("apply bn condition")
        averaged_model = condition_bn(
            config, averaged_model, dataloader, iterations=100
        )

        # evaluate averaged model
        results_step = evaluate_single_model(config, averaged_model)
        if alpha == 0:
            results = results_step
            example_key = list(results.keys())[0]
            results["alpha"] = [alpha for _ in range(len(results_step[example_key]))]
        else:
            for key in results.keys():
                results[key].append(results_step[key])
                # results[key].extend(results_step[key])
            results["alpha"].extend([alpha for _ in range(len(results_step[key]))])

    return results
