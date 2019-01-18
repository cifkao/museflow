import os
import random

import numpy as np
import tensorflow as tf
import yaml

from museflow import logger
from museflow.config import configurable, configure


@configurable()
class Model:
    """A base class for models."""

    def __init__(self, logdir, **kwargs):
        self._logdir = logdir
        self._args = kwargs

        tf.reset_default_graph()
        self._graph = tf.get_default_graph()

        seed = self._args.get('random_seed')
        if seed is not None:
            self._graph.seed = seed
            random.seed(seed)
            np.random.seed(seed)

    @classmethod
    def from_yaml(cls, logdir, config_file=None, **kwargs):
        """Construct the model from a given YAML config file.

        Args:
            logdir: The path to the log directory of the model.
            config_file: A path to a configuration file or an open file object. If not given,
                the file 'model.yaml' in the log directory will be used instead.
            **kwargs: Keyword arguments to the model's `__init__` method.
        Returns:
            The model object.
        """
        if config_file is None:
            config_file = os.path.join(logdir, 'model.yaml')

        if isinstance(config_file, str):
            with open(config_file, 'rb') as f:
                return cls.from_yaml(logdir, config_file=f, **kwargs)

        config_dict = yaml.load(config_file)
        logger.debug(config_dict)
        return configure(cls, config_dict, logdir=logdir, **kwargs)
