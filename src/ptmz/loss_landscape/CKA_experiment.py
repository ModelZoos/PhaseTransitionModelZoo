from ray.tune import Trainable
from ptmz.loss_landscape.CKA import compute_cka
import logging


###############################################################################
# define Tune Trainable
###############################################################################
class CKATrainable(Trainable):
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
        logging.info("Set up CKA Trainable")

        # set trainable properties
        self.config = config
        self.device = config.get("device", None)

    # step ####
    def step(self):
        # compute 1 step of results

        # compute CKA
        cka_results = compute_cka(self.config, self.device)

        return cka_results

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
