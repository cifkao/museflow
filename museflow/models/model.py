import random

import numpy as np
import tensorflow as tf

from museflow.config import Configurable


class Model(Configurable):
    """A base class for models."""

    def __init__(self, config=None, **kwargs):
        Configurable.__init__(self, config)
        self._args = kwargs

        tf.reset_default_graph()
        self._graph = tf.get_default_graph()

        seed = self._args.get('random_seed')
        if seed is not None:
            self._graph.seed = seed
            random.seed(seed)
            np.random.seed(seed)
