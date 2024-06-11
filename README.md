# Supplemantary Material To Phase Transition Model Zoos

[TODO](): Add overview figure
Dear Reviewers,

In this repository, we provide full code to reproduce the training, evaluation and loss-landscape metrics computation of the model zoos proposed in our submission. We also provide a notebook to reproduce the figures in the paper visualizing the different phases. 

[TODO](): Add phase figure

This repository contains three code sections: `src`, `scripts` and a notebook to visualize and explore the dataset.  

## Code
[TODO](): describe code structure
In `src`, we collect model definitions, trainer code, loss-landscape and model averaging code. Model definitions and runners are in `src/ptmz/models/`. Here, `def_resnet_width.py` contains the ResNet model class and training steps. `def_NN_experiment.py` contains code to run experiments over grids using [ray.tune](https://docs.ray.io/en/latest/tune/index.html). 
The loss landscape metrics are collected in `src/ptmz/loss_landscape/`. We include code to compute the hessian metrics and ray experiment in `hessian_metric.py` and `hessian_experiment.py`, with dependencies on `src/ptmz/loss_landscape/pyhessian`. The CKA implementation is likewise in `CKA.py` and `CKA_experiment.py` with dependencies on `CKA_utils.py`. Similarly, Bezier mode connectivity is implemented in `curve_fitting.py` and `curve_fitting_experiment.py` with utils in `curves.py`.
Code to evaluate phases in model averaging is in n`src/ptmz/model_averaging/`. Averaging one model over different training epochs is implemented in `average_model_epochs.py` and `average_model_epochs_experiment.py`. Averaging over different seeds is in `average_models.py` and `average_model_experiment.py`, with additional model aligenment in `align_models.py`.

[TODO](): should we discuss the pyhessian stuff somewhere / give reference to where that comes from?

## Experiment replication
We include all code to reproduce our experiments and re-train or extend our model zoos, or re-compute the loss-landscape metrics.
The experiment runner files can be fonud in `scripts` with one sub-directory per zoo. Within each subdirectory, you will find experiment runners to train the zoos, e.g.,`train_zoo_cifar10_resnet18_wide.py`, but also experiment runners to compute the loss landscape metrics and model averaging experiments.
To train the zoos, one has to adapt the paths to the local file structure, configure available GPUS, and then run `python3 train_zoo_cifar10_resnet18_wide.py`. This will configure the experiment space and run all experiments on the temperature-load grid with seed repetitions using [ray.tune](https://docs.ray.io/en/latest/tune/index.html). The experiments to compute loss-landscape metrics are run in a similar fashion. These require only the path to an existing trained population.

## Visualization of Phases
[TODO](): 
This repository also contains code to reproduce figures from the submission.   
[TODO]() should we add a dataframe here, so people can just plot the figures? Or should we (also) link to the zips so they can download and play around with them?

## Dataset Samples
[TODO](): fit in: 
Two zoo samples, one ResNet18 and ResNet50 zoo can be found here [TODO]().
Please note that the dataset format is restricted at this point to facilitate double blind reviewing. 
We purposefully leave the dataset relatively raw, since we do not want to restrict use cases. For the camera ready version, we will provide code to make the dataset accessible for the use-cases we can envision. We welcome feedback on how best to present these datasets.

### Dataset Directory Structure and Information 
[TODO](): describe directory structure.

#