from ray.tune import Trainable
from ptmz.model_averaging.interpolate_models import evaluate_model_interpolation
import logging


###############################################################################
# define Tune Trainable
###############################################################################
class ModelAverageTrainable(Trainable):
    """
    tune trainable wrapper around model average experiments
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
        # compute interpolations
        interpolation_results = evaluate_model_interpolation(self.config, self.device)

        return interpolation_results

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
