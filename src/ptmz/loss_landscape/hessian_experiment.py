from ray.tune import Trainable
from ptmz.loss_landscape.hessian_metric import compute_hessian
import logging


###############################################################################
# define Tune Trainable
###############################################################################
class HessianTrainable(Trainable):
    """
    tune trainable wrapper around AE model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Args:
        config (dict): config dictionary
    """

    def setup(self, config, data=None):
        """
        Init function to set up experiment. Configures data, augmentaion, and module
        Args:
            config (dict): config dictionary
            data (dict): data dictionary (optional)
        """
        logging.info("Set up Hessian Trainable")

        # set trainable properties
        self.config = config
        self.device = config.get("device", None)

    # step ####
    def step(self):
        # compute 1 step of results

        # compute hessian
        hessian_results = compute_hessian(self.config, self.device)

        return hessian_results

    def save_checkpoint(self, experiment_dir):
        """
        Not implemented - stateless experiment
        """
        # tune apparently expects to return the directory
        return experiment_dir

    def load_checkpoint(self, experiment_dir):
        """
        Not implemented - stateless experiment
        """
        # tune apparently expects to return the directory
        return experiment_dir
