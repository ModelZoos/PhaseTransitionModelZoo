import numpy as np
import torch
from ptmz.loss_landscape.pyhessian.hessian import hessian

import json
from pathlib import Path

from ptmz.sampling.load_dataset import load_datasets_from_config
from ptmz.models.def_net_width import NNmodule_width
from ptmz.models.def_net import NNmodule


def compute_hessian(config: dict, device: None, verbosity: int = 15) -> dict:
    """Computes Hessian for a list of models.

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

    print(f'compute Hessian on device "{device}"')
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
        config_dx["training::batchsize"] = config["hessian_batch_size"]

    # load data
    trainloader, testloader, valloader = load_datasets_from_config(configs[0])

    # else:
    dataloader_train = trainloader

    if testloader:
        dataloader_val = testloader
        results["dataloader_val"] = "test"
    elif valloader:
        dataloader_val = valloader
        results["dataloader_val"] = "val"
    else:
        dataloader_val = None

    # prepare train data
    print(f"prepare data")
    batch_num = config["hessian_number_batches"]
    if batch_num == 1:
        for inputs, labels in dataloader_train:
            hessian_dataloader_train = (inputs, labels)
            break
    else:
        hessian_dataloader_train = []
        for i, (inputs, labels) in enumerate(dataloader_train):
            hessian_dataloader_train.append((inputs, labels))
            if i == batch_num - 1:
                break
    if dataloader_val is not None:
        if batch_num == 1:
            for inputs, labels in dataloader_val:
                hessian_dataloader_val = (inputs, labels)
                break
        else:
            hessian_dataloader_val = []
            for i, (inputs, labels) in enumerate(dataloader_val):
                hessian_dataloader_val.append((inputs, labels))
                if i == batch_num - 1:
                    break

    # overide device
    for config_dx in configs:
        config_dx["device"] = device
        config_dx["cuda"] = device == torch.device("cuda")

    # instantiate model
    print(f"instantiate model")
    modules = [NNmodule_width(config_dx) for config_dx in configs]

    # load checkpoints
    print(f"load checkpoints")
    checkpoint_paths = [
        Path(mpdx).joinpath(f"checkpoint_{epoch:06d}", "checkpoints")
        for mpdx in config["model_paths"]
    ]
    checkpoints = [
        torch.load(check_path, map_location=device) for check_path in checkpoint_paths
    ]
    print(f"load state dicts")
    for module, checkpoint in zip(modules, checkpoints):
        clean_checkpoint = clean_state_dict(checkpoint)
        module.model.load_state_dict(clean_checkpoint)
        module.eval()

    # init outputs
    hessian_top_ev_train = []
    hessian_trace_train = []
    hessian_top_ev_val = []
    hessian_trace_val = []

    # iterate over models
    print(f"compute Hessian")
    for module in modules:
        # compute hessian
        module.to(device)
        if batch_num == 1:
            hessian_comp_train = hessian(
                model=module,
                criterion=module.criterion,
                data=hessian_dataloader_train,
                cuda=True if device == torch.device("cuda") else False,
            )
            if dataloader_val is not None:
                hessian_comp_val = hessian(
                    model=module,
                    criterion=module.criterion,
                    data=hessian_dataloader_val,
                    cuda=True if device == torch.device("cuda") else False,
                )
        else:
            hessian_comp_train = hessian(
                model=module,
                criterion=module.criterion,
                dataloader=hessian_dataloader_train,
                cuda=True if device == torch.device("cuda") else False,
            )
            if dataloader_val is not None:
                hessian_comp_val = hessian(
                    model=module,
                    criterion=module.criterion,
                    dataloader=hessian_dataloader_val,
                    cuda=True if device == torch.device("cuda") else False,
                )

        print("********** finish data londing and begin Hessian computation **********")

        top_eigenvalues_train, _ = hessian_comp_train.eigenvalues()
        trace_train = hessian_comp_train.trace()
        hessian_top_ev_train.append(top_eigenvalues_train)
        hessian_trace_train.append(np.mean(trace_train))
        if verbosity > 10:
            print("\n***Top Eigenvalues: ", top_eigenvalues_train)
            print("\n***Trace: ", np.mean(trace_train))

        # val
        if dataloader_val is not None:
            top_eigenvalues_val, _ = hessian_comp_val.eigenvalues()
            trace_val = hessian_comp_val.trace()
            hessian_top_ev_val.append(top_eigenvalues_val)
            hessian_trace_val.append(np.mean(trace_val))

    # train
    results["hessian_top_eigenvalues_train_list"] = hessian_top_ev_train
    results["hessina_trace_train_list"] = hessian_trace_train
    results["hessian_top_eigenvalues_train_mean"] = np.mean(
        np.array(hessian_top_ev_train)
    )
    results["hessian_trace_train_mean"] = np.mean(np.array(hessian_trace_train))
    # val
    if dataloader_val is not None:
        results["hessian_top_eigenvalues_val_list"] = hessian_top_ev_val
        results["hessina_trace_val_list"] = hessian_trace_val
        results["hessian_top_eigenvalues_val_mean"] = np.mean(
            np.array(hessian_top_ev_val)
        )
        results["hessian_trace_val_mean"] = np.mean(np.array(hessian_trace_val))

    # return results
    return results


def clean_state_dict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, params in state_dict.items():
        name = key.replace("module.", "")  # remove `module.`
        new_state_dict[name] = params
    return new_state_dict
