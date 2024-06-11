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


def evaluate_averaged_model_epochs(
    config: dict,
    device: torch.device = None,
    verbosity: int = 15,
) -> dict:
    """
    Averages the weights of multiple models and evaluates their performance.
    Args:
        config (dict): config dictionary. Must contain the following
            - model_paths (list): list of paths to model configs
            - epoch: epoch of checkpoint to load
            - align_models (bool): flag to align models
            - finetuning_epochs (int): number of finetuning epochs
        device (torch.device): device to use for computation
        verbosity (int): verbosity level
    Returns:
        Evaluation results
    """

    align_models_flag = config.get("align_models", True)
    finetuning_epochs = config.get("finetuning_epochs", 0)

    # extract checkpoint epoch list
    epoch_list = config["epoch_list"]

    # configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda" or device == torch.device("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print(f"Warning: device {device} is not available. Using CPU instead.")
            device = torch.device("cpu")

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
    checkpoint_paths = []
    for idx in range(len(config["model_paths"])):
        checkpoint_paths.append(
            [
                Path(config["model_paths"][idx]).joinpath(
                    f"checkpoint_{edx:06d}", "checkpoints"
                )
                for edx in epoch_list
            ]
        )
    # load checkpoints
    print("load checkpoints")
    load_device = "cpu"
    checkpoints = []
    for idx in range(len(checkpoint_paths)):
        checks_tmp = [
            torch.load(pdx, map_location=load_device) for pdx in checkpoint_paths[idx]
        ]
        # rename checkpoint keys
        checks_tmp = [rename_state_keys(check, "module.", "") for check in checks_tmp]
        # append
        checkpoints.append(checks_tmp)

    if verbosity > 10:
        print(f"number of models: {len(checkpoints)}")
        print(f"number of epochs per model: {len(checkpoints[0])}")

    # align models
    if align_models_flag:
        print(f"align models")
        for idx in range(len(checkpoints)):
            checkpoints[idx] = align_models(configs[0], checkpoints[idx])

    # average models
    print(f"average models")
    averaged_models = []
    for idx in range(len(checkpoints)):
        averaged_models.append(
            average_model_checkpoints(
                checkpoints[idx], [1 / len(checkpoints[idx])] * len(checkpoints[idx])
            )
        )
    # apply batch norm conditioning
    print("apply bn condition")
    for idx in range(len(averaged_models)):
        averaged_models[idx] = condition_bn(
            configs[0], averaged_models[idx], trainloader, iterations=100
        )

    results = {
        "individual": [],
    }
    for idx in range(len(averaged_models)):
        # evaluate averaged model
        res_tmp = evaluate_single_model(
            configs[0], averaged_models[idx], finetuning_epochs
        )
        res_tmp["model_path"] = config["model_paths"][idx]
        results["individual"].append(res_tmp)

    # aggregate numerical values in dict of lists in average
    results["aggregated"] = {}
    results["aggregated"]["epoch_list"] = epoch_list
    # iterate over output keys
    for key in [
        "loss_train",
        "loss_val",
        "loss_test",
        "acc_train",
        "acc_val",
        "acc_test",
    ]:
        if key in results["individual"][0]:
            results["aggregated"][key] = []
            for idx in range(len(results["individual"])):
                results["aggregated"][key].append(results["individual"][idx][key])
            # compute mean per entry so that we have a list of mean entries
            results["aggregated"][key] = np.mean(results["aggregated"][key], axis=0)

    return results


def rename_state_keys(state, key_old, key_new):
    new_state = OrderedDict()

    for key in state.keys():
        nkey = key.replace(key_old, key_new)
        new_state[nkey] = state[key]
    return new_state
