# Description: combines multiple models into a single model

import torch
from typing import List
from collections import OrderedDict
from copy import deepcopy


def average_model_checkpoints(
    checkpoint_list: List[OrderedDict], weight_list: List[float]
) -> OrderedDict:
    """
    Average the weight_list of multiple models.
    Args:
        checkpoint_list: List of model checkpoints
        weight_list: List of weight_list for each model
    Returns:
        Averaged model
    """
    assert abs(sum(weight_list) - 1) < 1e-4, "Weight_list must sum to 1"
    assert len(checkpoint_list) == len(
        weight_list
    ), "Number of models and weight_list must match"
    assert len(checkpoint_list) > 0, "At least one model must be provided"
    assert all(
        check.keys() == checkpoint_list[0].keys() for check in checkpoint_list
    ), "Models must have the same keys"
    assert all(
        check[key].shape == checkpoint_list[0][key].shape
        for check in checkpoint_list
        for key in checkpoint_list[0].keys()
    ), "Models must have the same shapes"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    averaged_model = deepcopy(checkpoint_list[0])
    for key in averaged_model.keys():
        # skip non weight / bias parameters
        if not "weight" in key or "bias" in key:
            continue
        averaged_model[key] = torch.zeros_like(averaged_model[key]).to(device)
        for check, weight in zip(checkpoint_list, weight_list):
            try:
                averaged_model[key] += check[key].to(device) * weight
            except Exception as e:
                print(f"Error: {e}")
                print(f"key: {key}")
                print(f"check[key].shape: {check[key].shape}")
                print(f"check[key].device: {check[key].device}")
                print(f"averaged_model[key].shape: {averaged_model[key].shape}")
                print(f"averaged_model[key].device: {averaged_model[key].device}")
                raise e
    return averaged_model
