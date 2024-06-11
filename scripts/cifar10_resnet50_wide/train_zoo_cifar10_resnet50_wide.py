import os

# set environment variables to limit cpu usage
# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"

import sys
from pathlib import Path

import json

import ray
from ray import tune

# # from ray.tune.logger import DEFAULT_LOGGERS
# from ray.tune.logger import JsonLogger, CSVLogger
# from ray.tune.integration.wandb import WandbLogger
import argparse
import torch
from torchvision import datasets, transforms

from ptmz.models.def_NN_experiment import NN_tune_trainable


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--path-root", type=str, default="/ds2/model_zoos/taxonomy/cifar_10/", help=""
)
parser.add_argument(
    "--data-path",
    type=str,
    default="/ds2/model_zoos/taxonomy/cifar_10/",
    help="",
)
parser.add_argument("--epoch", type=int, default=150, help="")
parser.add_argument("--lr-list", type=float, nargs="+", default=[0.1], help="")
parser.add_argument(
    "--weightdecay-list", type=float, nargs="+", default=[5e-4], help=""
)
parser.add_argument(
    "--batchsize-list",
    type=int,
    nargs="+",
    default=[1024, 512, 256, 128, 64, 32, 16, 8],
    help="",
)
parser.add_argument(
    "--width-list",
    type=int,
    nargs="+",
    default=[2, 4, 8, 16, 32, 64, 128, 256],
    help="",
)
parser.add_argument("--seed-list", type=int, nargs="+", default=[0, 1, 2], help="")

args = parser.parse_args()

PATH_ROOT = Path(args.path_root)
DATA_PATH = args.data_path
NUM_EPOCH = args.epoch
lr_list = args.lr_list
weightdecay_list = args.weightdecay_list
batchsize_list = args.batchsize_list
seeds = args.seed_list  # list(range(0, 5))
width_list = args.width_list  # , steropes
model_type = "Resnet50_width"
cpus = 96
gpus = 8
cpu_per_trial = int(cpus // gpus)
gpu_fraction = 1


def main():
    # ray init to limit memory and storage
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}
    # experiment name
    experiment_name = f"tune_zoo_cifar-10_{model_type}_kaiming_uniform_{NUM_EPOCH}eps_width_{width_list[0]}-{width_list[-1]}"

    # set module parameters
    config = {}

    config["model::type"] = model_type
    config["model::channels_in"] = 3
    config["model::o_dim"] = 10
    config["model::nlin"] = "relu"
    config["model::dropout"] = 0.0
    config["model::init_type"] = "kaiming_uniform"
    config["model::use_bias"] = False
    config["model::width"] = tune.grid_search(width_list)
    config["optim::optimizer"] = "sgd"
    config["optim::lr"] = tune.grid_search(lr_list)
    config["optim::wd"] = tune.grid_search(weightdecay_list)
    config["optim::momentum"] = 0.9
    config["optim::scheduler"] = "OneCycleLR"
    # config["training::loss"] = "nll"
    config["training::loss"] = "crossentropy"
    config["training::dataloader"] = "normal"
    config["trainloader::workers"] = 6
    config["testloader::workers"] = 4

    # set seeds for reproducibility

    config["seed"] = tune.grid_search(seeds)

    # set training parameters
    net_dir = PATH_ROOT.joinpath(f"zoos/Cifar-10/{config['model::type']}_v2")
    try:
        net_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    print(f"Zoo directory: {net_dir.absolute()}")

    config["training::batchsize"] = tune.grid_search(batchsize_list)
    config["training::epochs_train"] = NUM_EPOCH
    config["training::output_epoch"] = 1  # keep every epoch, clean up later #5
    config["training::data_path"] = DATA_PATH

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False

    data_path = Path(DATA_PATH)

    # # normalization computed with:
    if not os.path.exists(data_path.joinpath("dataset_preprocessed.pt")):
        cifar_path = data_path.joinpath("image_data/CIFAR10")
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        trainset = datasets.CIFAR10(
            root=cifar_path,
            train=True,
            transform=train_transforms,
            download=True,
        )

        testset = datasets.CIFAR10(
            root=cifar_path,
            train=False,
            transform=test_transforms,
            download=True,
        )

        # save dataset and seed in data directory
        dataset = {
            "trainset": trainset,
            "testset": testset,
        }
        torch.save(dataset, data_path.joinpath("dataset_preprocessed.pt"))

    # # save datasets in zoo directory
    config["dataset::dump"] = data_path.joinpath("dataset_preprocessed.pt").absolute()

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )

    # save config as json file
    net_dir.joinpath(experiment_name).mkdir(parents=True, exist_ok=True)
    with open((net_dir.joinpath(experiment_name, "config.json")), "w") as f:
        json.dump(config, f, default=str)

    # generate empty readme.md file   ?? maybe create and copy template
    # check if readme.md exists
    readme_file = net_dir.joinpath(experiment_name, "readme.md")
    if readme_file.is_file():
        pass
    # if not, make empty readme
    else:
        with open(readme_file, "w") as f:
            pass
    ray.shutdown()
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

    print(net_dir)
    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=NN_tune_trainable,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=net_dir,
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


if __name__ == "__main__":
    main()
