from ray.tune import Trainable
from ptmz.model_averaging.average_model_epochs import evaluate_averaged_model_epochs
import logging


###############################################################################
# define Tune Trainable
###############################################################################
class ModelAverageEpochsTrainable(Trainable):
    """
    tune trainable wrapper around model average experiments - averaging over epoch_list
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
        logging.info("Set up model averaging trainable")

        # set trainable properties
        self.config = config
        self.device = config.get("device", None)

    # step ####
    def step(self):
        # compute 1 step of results

        # compute averagings
        averaging_epochs_results = evaluate_averaged_model_epochs(
            self.config, self.device
        )

        return averaging_epochs_results

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
