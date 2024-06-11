# code to evaluate a single model
import torch
import copy
from typing import List
from collections import OrderedDict
from rebasin import PermutationCoordinateDescent
from ptmz.model_averagingload_dataset import load_datasets_from_config
from ptmz.models.def_net import NNmodule
import copy


def align_models(
    config: dict,
    checkpoints: List[OrderedDict],
) -> List[OrderedDict]:
    """
    Aligns the weights of multiple models.
    Args:
        config: Configuration dictionary
        checkpoints: List of model checkpoints
    Returns:
        Aligned models
    """
    assert len(checkpoints) > 1, "At least two models must be provided"
    assert all(
        check.keys() == checkpoints[0].keys() for check in checkpoints
    ), "Models must have the same keys"
    assert all(
        check[key].shape == checkpoints[0][key].shape
        for check in checkpoints
        for key in checkpoints[0].keys()
    ), "Models must have the same shapes"

    # instantiate models
    config = copy.deepcopy(config)
    # config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = "cpu"
    config["scheduler::steps_per_epoch"] = 123
    module_a = NNmodule(config)
    module_b = NNmodule(config)

    # fix naming
    def rename_state_keys(state, key_old, key_new):
        new_state = OrderedDict()

        for key in state.keys():
            nkey = key.replace(key_old, key_new)
            new_state[nkey] = state[key]
        return new_state

    checkpoints = [rename_state_keys(check, "module.", "") for check in checkpoints]
    # load checkpoint 1
    module_a.model.load_state_dict(checkpoints[0])

    # get dataset
    trainloader, testloader, valloader = load_datasets_from_config(config)
    # get images from first batch
    input_data = next(iter(trainloader))[0]

    aligned_checkpoints = []
    # add first checkpoint as reference
    aligned_checkpoints.append(checkpoints[0])
    # iterate over remaining checkpoints
    for check in checkpoints[1:]:
        # make copy of checkpoint
        aligned_check = copy.deepcopy(check)
        # load checkpoint on module_b
        module_b.model.load_state_dict(aligned_check)
        # send models to device
        module_a.model = module_a.model.to(config["device"])
        module_b.model = module_b.model.to(config["device"])
        # Rebasin
        pcd = PermutationCoordinateDescent(
            module_a.model,
            module_b.model,
            input_data,
            device_a=config["device"],
            device_b=config["device"],
        )  # weight-matching
        pcd.rebasin()  # Rebasin model_b towards model_a. Automatically updates model_b
        # extract state_dict from module_b
        aligned_checkpoints.append(module_b.model.state_dict())

    # return aligned checkpoints
    return aligned_checkpoints
