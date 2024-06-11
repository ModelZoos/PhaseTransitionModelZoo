import os

# set environment variables to limit cpu usage
# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

from pathlib import Path

import json

import ray
from ray import tune

import torch

from ptmz.loss_landscape.CKA_experiment import CKATrainable

import numpy as np

import pandas as pd

from tqdm import tqdm

cpus = 96
gpus = 8
cpu_per_trial = int(cpus // gpus)
gpu_fraction = 1
model_type = "Resnet18_width"
# NUM_EPOCH = 150
NUM_EPOCH = 135
width_list = [2, 4, 8, 16, 32, 64, 128, 256]
PATH_ROOT = Path("/ds2/model_zoos/taxonomy/svhn/zoos/SVHN/Resnet18_width_v2")


def main():
    # ray init to limit memory and storage
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}
    # experiment name
    experiment_name = f"CKA_svhn_{model_type}_kaiming_uniform_{NUM_EPOCH}eps_width_{width_list[0]}-{width_list[-1]}"
    experiment_dir = PATH_ROOT

    # set training parameters
    root = Path(
        "/ds2/model_zoos/taxonomy/svhn/zoos/SVHN/Resnet18_width_v2/tune_zoo_SVHN_Resnet18_width_kaiming_uniform_150eps_width_2-256"
    )

    # load configs to dataframe
    config_df = compile_configs(root)

    # create experiment list
    experiments = create_experiment_list(
        model_dataframe=config_df,
        vary_keys=["seed"],
        match_keys=["training::batchsize", "model::width"],
    )
    config = {
        "model_paths": tune.grid_search(experiments),
        "epoch": NUM_EPOCH,
        "CKA_repeats": 1,
        "CKA_num_batches": 10,
        "CKA_batch_size": 128,
        "flattenHW": True,
    }

    experiment_dir.joinpath(experiment_name).mkdir(parents=True, exist_ok=True)

    assert ray.is_initialized() == False

    config["resources"] = resources_per_trial
    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=True,
        dashboard_host="0.0.0.0",  # 0.0.0.0 is the host of the docker, (localhost is the container) (https://github.com/ray-project/ray/issues/11457#issuecomment-1325344221)
        dashboard_port=8265,
    )
    assert ray.is_initialized() == True

    print(f"started ray. running dashboard under {context.dashboard_url}")

    print(experiment_dir)
    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=CKATrainable,
        stop={
            "training_iteration": 1,  # just the one epoch
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=1,
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=experiment_dir,
        resources_per_trial=resources_per_trial,
    )

    # run
    ray.tune.run_experiments(
        experiments=experiment,
        # resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        # resume=True,  # resumes from previous run. if run should be done all over, set resume=False
        # resume="ERRORED_ONLY",  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        verbose=3,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


def compile_configs(root_dir):
    # Create a Path object for the root directory
    root_path = Path(root_dir)

    # List to hold all the configurations and their paths
    data = []

    # Walk through the directory structure
    path_list = [pdx for pdx in root_path.iterdir() if pdx.is_dir()]
    for pdx in tqdm(path_list):
        json_file = pdx.joinpath("params.json")
        # Load the JSON file
        with json_file.open("r") as f:
            config = json.load(f)

        # Append the configuration and the path to the data list
        config["file_path"] = str(json_file).replace("params.json", "")
        data.append(config)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    return df


def create_experiment_list(
    model_dataframe,
    vary_keys=["seed"],
    match_keys=["training::batchsize", "model::width"],
):
    """
    Create a list of experiments to run.

    Args:
        model_dataframe (pd.DataFrame): dataframe containing the model configurations
        vary_keys (list): list of keys to vary in the experiments
        match_keys (list): list of keys to match in the experiments

    Returns:
        list: list of experiments to run
    """

    # Check if input is a DataFrame
    if not isinstance(model_dataframe, pd.DataFrame):
        raise ValueError("The model_dataframe should be a pandas DataFrame.")

    # Check if all vary_keys and match_keys are in the DataFrame columns
    for key in vary_keys + match_keys:
        if key not in model_dataframe.columns:
            raise ValueError(f"Key '{key}' not found in DataFrame columns.")

    # Group the dataframe by the match_keys
    grouped = model_dataframe.groupby(match_keys)

    experiments = []

    # Iterate over each group
    for _, group in grouped:
        # Convert the entire group to a list of dictionaries
        group_experiments = group.to_dict(orient="records")
        experiments.append([gdx["file_path"] for gdx in group_experiments])

    return experiments


if __name__ == "__main__":
    main()
