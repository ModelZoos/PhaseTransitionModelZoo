from __future__ import print_function


import json
from pathlib import Path
import numpy as np
import torch

from ptmz.sampling.load_dataset import load_datasets_from_config
from ptmz.models.def_net_width import NNmodule_width
from ptmz.models.def_net import NNmodule


# from utils import *
from ptmz.loss_landscape.CKA_utils import (
    feature_space_linear_cka,
    cka_compute,
    gram_linear,
)


def compute_cka(config: dict, device: None, verbosity: int = 15) -> dict:
    """Computes CKA similarity for two models.

    Args:
        config (dict): config dictionary. Must containn the following
            - model_paths (list): list of paths to model configs
            - epoch: epoch of checkpoint to load
        device (torch.device): device to use for computation
        verbosity (int): verbosity level
    Returns:
        dict: results dictionary
    """
    # configure device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda" or device == torch.device("cuda"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print(f"Warning: device {device} is not available. Using CPU instead.")
            device = torch.device("cpu")

    # init output
    results = {}

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
        config_dx["training::batchsize"] = config["CKA_batch_size"]

    # load data
    trainloader, testloader, valloader = load_datasets_from_config(configs[0])

    if valloader:
        dataloader = valloader
        results["dataloader"] = "val"
    elif testloader:
        dataloader = testloader
        results["dataloader"] = "test"
    else:
        dataloader = trainloader
        results["dataloader"] = "train"

    # instantiate model
    modules = [NNmodule_width(config_dx) for config_dx in configs]

    # load checkpoints
    # replace "params.json" with f"checkpoint_000{epoch:06d}/checkpoints" to get the checkpoint path
    checkpoint_paths = [
        Path(mpdx).joinpath(f"checkpoint_{epoch:06d}", "checkpoints")
        for mpdx in config["model_paths"]
    ]
    checkpoints = [
        torch.load(check_path, map_location=device) for check_path in checkpoint_paths
    ]
    for module, checkpoint in zip(modules, checkpoints):
        clean_checkpoint = clean_state_dict(checkpoint)
        module.model.load_state_dict(clean_checkpoint)
        module.eval()

    # compute CKA
    CKA_repeats = config["CKA_repeats"]
    CKA_num_batches = config["CKA_num_batches"]
    flattenHW = config["flattenHW"]
    # init ouptut
    cka_from_features_pairs = []
    # iterate over outer model
    for idx, module_1 in enumerate(modules):
        module_1.model.to(device)
        # get second model
        for jdx, module_2 in enumerate(modules[idx + 1 :]):
            module_2.model.to(device)
            # init output for current pair
            cka_from_features_average = []
            # iterate over repeats
            for CKA_repeat_runs in range(CKA_repeats):
                # init output for these features
                cka_from_features = []
                # compute predictions - we only consider logits for CKA. Intermediate representations require hooks
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    if batch_idx == CKA_num_batches:
                        break
                    # compute predictions
                    with torch.no_grad():
                        inputs = inputs.to(device)
                        outputs_1 = module_1.model(inputs)
                        outputs_2 = module_2.model(inputs)
                        outputs_1 = outputs_1.to("cpu")
                        outputs_2 = outputs_2.to("cpu")
                    # compute CKA similarity
                    print(outputs_1.shape)
                    if flattenHW:
                        cka_from_features.append(
                            feature_space_linear_cka(
                                outputs_1.detach().numpy(), outputs_2.detach().numpy()
                            )
                        )
                    else:
                        cka_from_features.append(
                            cka_compute(
                                gram_linear(outputs_1.numpy()),
                                gram_linear(outputs_2.numpy()),
                            )
                        )
                # append results over batches
                cka_from_features_average.append(cka_from_features)
            # compute mean for current pair
            cka_from_features_average = np.mean(
                np.array(cka_from_features_average), axis=0
            )
            # append current mean to pairs value
            cka_from_features_pairs.append(cka_from_features_average)

            # print results

    cka_pairs_average = np.mean(np.array(cka_from_features_pairs), axis=0)

    if verbosity > 10:
        print("cka_from_features shape")
        print(len(cka_pairs_average))
        print(cka_pairs_average)
        print(len(cka_from_features_pairs))
        print(cka_from_features_pairs)

    # store results
    results["representation_similarity"] = cka_pairs_average
    results["representation_similarity_pairs"] = cka_from_features_pairs

    # return
    return results


def clean_state_dict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, params in state_dict.items():
        name = key.replace("module.", "")  # remove `module.`
        new_state_dict[name] = params
    return new_state_dict
