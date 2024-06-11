import numpy as np
import torch

import json
from pathlib import Path

from ptmz.sampling.load_dataset import load_datasets_from_config

import os
import torch.nn.functional as F
import copy

from ptmz.loss_landscape.utils import test_acc_loss
from tqdm import tqdm

from ptmz.loss_landscape import curves
from ptmz.models.def_net_width import (
    ResNet18_width,
    ResNet34_width,
    ResNet50_width,
    ResNet101_width,
    ResNet152_width,
)
from ptmz.models.def_resnet_width_curves import (
    ResNet18Curve,
    ResNet34Curve,
    ResNet50Curve,
    ResNet101Curve,
    ResNet152Curve,
)


def compute_curves(config: dict, device: None, verbosity: int = 15) -> dict:
    """computes bezier curves"""
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
    curve_bs = config["curve::batchsize"]

    # extract model config
    config_paths = [
        Path(mpdx).joinpath("params.json") for mpdx in config["model_paths"]
    ]
    configs = [json.load(Path(config_pdx).open("r")) for config_pdx in config_paths]
    for config_dx in configs:
        config_dx["device"] = device
        config_dx["cuda"] = device == torch.device("cuda")
        config_dx["scheduler::steps_per_epoch"] = 123
        config_dx["training::batchsize"] = curve_bs

    # load data
    trainloader, testloader, valloader = load_datasets_from_config(configs[0])

    # load checkpoints
    checkpoint_paths = [
        Path(mpdx).joinpath(f"checkpoint_{epoch:06d}", "checkpoints")
        for mpdx in config["model_paths"]
    ]
    # configure Curves
    curve = getattr(curves, "Bezier")
    # infer curve architecture
    base_arch, curve_arch = get_curve_model(configs[0]["model::type"])
    # get config for curve
    num_points = config["curve::num_points"]
    num_bends = config["curve::num_bends"]
    fix_start = config["curve::fix_start"]
    fix_end = config["curve::fix_end"]

    config_curve = {
        "model::curve": curve,
        "model::curve_arch": curve_arch,
        "model::num_points": num_points,
        "model::num_bends": num_bends,
        "model::fix_start": fix_start,
        "model::fix_end": fix_end,
        "training::lr": config.get("curve::lr", 0.1),
        "training::weight_decay": config.get("curve::weight_decay", 0.1),
        "training::momentum": config.get("curve::momentum", 0.9),
        "training::epochs": config.get("curve::epochs", 10),
    }
    print(f"fit curve with config: {config_curve}")
    config_curve["training::trainloader"] = trainloader

    if testloader is not None:
        config_curve["training::testloader"] = testloader
    elif valloader is not None:
        config_curve["training::testloader"] = valloader
    else:
        raise ValueError("No test or validation set provided")

    config_base = configs[0]
    config_base["base_arch"] = base_arch

    results = {
        "individual": [],
    }
    # iterate over model pairs
    for idx, path_1 in enumerate(checkpoint_paths):
        # start start to path 1
        config_curve["model::init_start"] = path_1
        # iterate over remaining models
        for jdx, path_2 in enumerate(checkpoint_paths[idx + 1 :]):
            # set path_2 to curve end
            config_curve["model::init_end"] = path_2
            # fit curve
            res_tmp = compute_curve_pair(
                config_base=config_base,
                config_curve=config_curve,
                device=device,
            )
            results["individual"].append(res_tmp)

    # aggregate numerical values in dict of lists in average
    results["aggregated"] = {}
    # iterate over output keys
    for key in ["train", "eval"]:
        results["aggregated"][key] = {}
        # iterate over entries of output keys
        for key2 in results["individual"][0][key].keys():
            results["aggregated"][key][key2] = []
            for idx in range(len(results["individual"])):
                results["aggregated"][key][key2].append(
                    results["individual"][idx][key][key2]
                )
            # compute mean per entry so that we have a list of mean entries
            results["aggregated"][key][key2] = np.mean(
                results["aggregated"][key][key2], axis=0
            )
    # aggregate stats
    results["aggregated"]["stats"] = {}
    for key in ["train", "eval"]:
        results["aggregated"]["stats"][key] = {}
        for key2 in results["individual"][0]["stats"][key].keys():
            results["aggregated"]["stats"][key][key2] = []
            for idx in range(len(results["individual"])):
                results["aggregated"]["stats"][key][key2].append(
                    results["individual"][idx]["stats"][key][key2]
                )
            results["aggregated"]["stats"][key][key2] = np.mean(
                results["aggregated"]["stats"][key][key2], axis=0
            )
    # aggregate mc
    results["aggregated"]["mc"] = {}
    for key in ["train_loss", "test_loss", "train_error", "test_error"]:
        results["aggregated"]["mc"][key] = []
        for idx in range(len(results["individual"])):
            results["aggregated"]["mc"][key].append(
                results["individual"][idx]["mc"][key]
            )
        results["aggregated"]["mc"][key] = np.mean(
            results["aggregated"]["mc"][key], axis=0
        )

    return results


def compute_curve_pair(
    config_base,
    config_curve,
    device,
):
    """
    Fits low-loss curve between two models with # bends in between.
    Subsequently evaluates the interpolation along that curve.

    Args:
        config_base: base model configuration
        config_curve: curve model configuration
        device: device to use
    """

    curve = config_curve["model::curve"]
    curve_arch = config_curve["model::curve_arch"]
    num_points = config_curve["model::num_points"]
    num_bends = config_curve["model::num_bends"]
    fix_start = config_curve["model::fix_start"]
    fix_end = config_curve["model::fix_end"]
    init_start = config_curve["model::init_start"]
    init_end = config_curve["model::init_end"]
    lr = config_curve.get("training::lr", 0.1)
    weight_decay = config_curve["training::weight_decay"]
    momentum = config_curve["training::momentum"]
    epochs = config_curve["training::epochs"]
    train_loader = config_curve["training::trainloader"]
    test_loader = config_curve["training::testloader"]
    base_arch = config_base["base_arch"]
    arch_kwargs = {"width": config_base["model::width"]}

    # get curve model
    model = curves.CurveNet(
        num_classes=config_base["model::o_dim"],
        curve=curve,
        architecture=curve_arch,
        num_bends=num_bends,
        fix_start=fix_start,
        fix_end=fix_end,
        architecture_kwargs=arch_kwargs,
    )

    base_model = None
    for path, k in [(init_start, 0), (init_end, num_bends - 1)]:
        if path:
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint = clean_state_dict(checkpoint)
            print("Loading %s as point #%d" % (path, k))

            base_model = base_arch(
                channels_in=config_base["model::channels_in"],
                out_dim=config_base["model::o_dim"],
                nlin=config_base["model::nlin"],
                dropout=config_base["model::dropout"],
                init_type=config_base["model::init_type"],
                width=config_base["model::width"],
            )
            base_model.load_state_dict(checkpoint)
            model.import_base_parameters(base_model, k)
        model.init_linear()

    model = model.to(device)
    criterion = F.cross_entropy  # check if this always holds (should for now..)
    regularizer = None if curve is None else curves.l2_regularizer(weight_decay)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay if curve is None else 0.0,
    )

    perf_train = {}
    start_epoch = 1
    columns = ["ep", "lr", "tr_loss", "tr_err", "te_nll", "te_err"]
    for key in columns:
        perf_train[key] = []
    has_bn = check_bn(model)
    test_res = {"loss": None, "accuracy": None, "nll": None}
    print("start training - ....")
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        lr = learning_rate_schedule(lr, epoch, epochs)
        adjust_learning_rate(optimizer, lr)
        print(f"start training...")
        train_res = train(train_loader, model, optimizer, criterion, regularizer)
        print(f"training epoch finished..")
        print(f"start testing..")
        test_res = test_acc_loss(test_loader, model, criterion, regularizer)
        print(f"testing done..")

        perf_train["ep"].append(epoch)
        perf_train["lr"].append(lr)
        perf_train["tr_loss"].append(train_res["loss"])
        perf_train["tr_err"].append(100 - train_res["accuracy"])
        perf_train["te_nll"].append(test_res["nll"])
        perf_train["te_err"].append(100 - test_res["accuracy"])

    # Part 2: evaluate interpolation along the bezier curve
    # export model state
    model_state = model.state_dict()

    # re-initialize model
    model = curves.CurveNet(
        num_classes=config_base["model::o_dim"],
        curve=curve,
        architecture=curve_arch,
        num_bends=num_bends,
        fix_start=fix_start,
        fix_end=fix_end,
        architecture_kwargs=arch_kwargs,
    )

    model.cuda()
    model.load_state_dict(model_state)

    criterion = F.cross_entropy
    regularizer = curves.l2_regularizer(weight_decay)

    # init interpolation values
    T = num_points
    ts = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_nll = np.zeros(T)
    tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_nll = np.zeros(T)
    te_acc = np.zeros(T)
    tr_err = np.zeros(T)
    te_err = np.zeros(T)
    dl = np.zeros(T)

    previous_weights = None

    # init outputs
    columns = [
        "t",
        "Train loss",
        "Train nll",
        "Train error (%)",
        "Test nll",
        "Test error (%)",
    ]
    perf_eval = {}
    for key in columns:
        perf_eval[key] = []

    t = torch.FloatTensor([0.0]).cuda()

    # interpolate
    for i, t_value in enumerate(ts):
        print(f"Interpolating at t={t.item()} - {i}/{t.shape[0]}...")
        t.data.fill_(t_value)
        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()
        print(f"updating bn at t={t.item()}")
        update_bn(train_loader, model, t=t)
        # break
        print(f"compute train perf at t={t.item()}")
        tr_res = test_acc_loss(train_loader, model, criterion, regularizer, t=t)
        print(f"compute test perf at t={t.item()}")
        te_res = test_acc_loss(test_loader, model, criterion, regularizer, t=t)

        tr_loss[i] = tr_res["loss"]
        tr_nll[i] = tr_res["nll"]
        tr_acc[i] = tr_res["accuracy"]
        tr_err[i] = 100.0 - tr_acc[i]
        te_loss[i] = te_res["loss"]
        te_nll[i] = te_res["nll"]
        te_acc[i] = te_res["accuracy"]
        te_err[i] = 100.0 - te_acc[i]

        values = [t.item(), tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]

        for key, value in zip(columns, values):
            perf_eval[key].append(value)

    # compute stats
    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = compute_stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = compute_stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = compute_stats(tr_err, dl)

    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = compute_stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = compute_stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = compute_stats(te_err, dl)

    # compute mc
    tr_loss_mc = compute_mc(tr_loss, mode="min")
    te_loss_mc = compute_mc(te_loss, mode="min")
    tr_err_mc = compute_mc(tr_err, mode="min")
    te_err_mc = compute_mc(te_err, mode="min")

    stats = {
        "train": {
            "loss": [tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
            "nll": [tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int],
            "error": [tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        },
        "eval": {
            "loss": [te_loss_min, te_loss_max, te_loss_avg, te_loss_int],
            "nll": [te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
            "error": [te_err_min, te_err_max, te_err_avg, te_err_int],
        },
        "legend": ["min", "max", "avg", "int"],
    }

    # join in output
    perf = {
        "train": perf_train,
        "eval": perf_eval,
        "stats": stats,
        "mc": {
            "train_loss": tr_loss_mc,
            "test_loss": te_loss_mc,
            "train_error": tr_err_mc,
            "test_error": te_err_mc,
        },
        "config_curve": config_curve,
    }
    # return
    return perf


def compute_stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


def compute_mc(values, mode="min"):
    """
    Computes mode connectivity of a given curve.
    Difference between the average of the first and last value and the best value in between.

    Args:
        values: list of values
        mode: mode of the curve (min or max)

    Returns:
        mode connectivity

    """
    # u = np.argmax(  np.abs(result - (result[0] + result[-1]) /2) )
    # gap = result[u] - ((result[0] + result[-1]) /2 )
    if mode == "max":
        # accuracy
        idx_max = np.argmax(np.abs(values - 0.5 * (values[0] + values[-1])))
        # positive values -> acc on the ends is lower than in the middle (under-trained)
        # negative values -> acc in the middle is lower than on the ends (barrier)
        mc = values[idx_max] - 0.5 * (values[0] + values[-1])
    elif mode == "min":
        # loss / error
        idx_max = np.argmax(np.abs(values - 0.5 * (values[0] + values[-1])))
        # positive values -> loss on the ends is higher than in the middle (under-trained)
        # negative values -> loss in the middle is higher than on the ends (barrier)
        mc = 0.5 * (values[0] + values[-1]) - values[idx_max]
    return mc


def clean_state_dict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for key, params in state_dict.items():
        name = key.replace("module.", "")  # remove 'module.'
        new_state_dict[name] = params
    return new_state_dict


def lerp(lam, t1, t2):
    t3 = copy.deepcopy(t2)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def train(
    train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None
):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)

    tbar = tqdm(total=num_iters)
    model.train()
    print(f"start training epoch")
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        # print('model time', model_time)
        # import pdb; pdb.set_trace()
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('opt time', time.time() - before_opt)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()
        # print('batch time', time.time() - batch_start)
        tbar.update(1)
    return {
        "loss": loss_sum / len(train_loader.dataset),
        "accuracy": correct * 100.0 / len(train_loader.dataset),
    }


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {
        "epoch": epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)
    return filepath


def isbatchnorm(module):
    return issubclass(
        module.__class__, torch.nn.modules.batchnorm._BatchNorm
    ) or issubclass(module.__class__, curves._BatchNorm)


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    tbar = tqdm(total=len(loader), desc="update bn")
    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size
        tbar.update(1)

    model.apply(lambda module: _set_momenta(module, momenta))


def get_curve_model(model_type):
    if "Resnet18_width" in model_type:
        return ResNet18_width, ResNet18Curve
    elif "Resnet34_width" in model_type:
        return ResNet34_width, ResNet34Curve
    elif "Resnet50_width" in model_type:
        return ResNet50_width, ResNet50Curve
    elif "Resnet101_width" in model_type:
        return ResNet101_width, ResNet101Curve
    elif "Resnet152_width" in model_type:
        return ResNet152_width, ResNet152Curve
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
