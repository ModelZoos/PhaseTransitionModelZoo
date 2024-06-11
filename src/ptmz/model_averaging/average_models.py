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


def evaluate_averaged_models(
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

    # average models
    print(f"average models")
    averaged_model = average_model_checkpoints(
        checkpoints, [1 / len(checkpoints)] * len(checkpoints)  # uniform weights
    )
    # apply batch norm conditioning
    print("apply bn condition")
    averaged_model = condition_bn(
        configs[0], averaged_model, trainloader, iterations=100
    )

    # evaluate averaged model
    print(f"evaluate averaged model")
    results = evaluate_single_model(configs[0], averaged_model, finetuning_epochs)
    return results
