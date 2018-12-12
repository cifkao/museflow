import tensorflow as tf

from museflow.config import Configurable
from .component import Component, using_scope


class EmbeddingLayer(Component, Configurable):

    def __init__(self, input_size, output_size, name='embedding', config=None):
        Component.__init__(self, name=name)
        Configurable.__init__(self, config)

        self.input_size = input_size
        self.output_size = output_size

        with self.use_scope():
            self.embedding_matrix = tf.get_variable(
                'embedding_matrix', shape=[self.input_size, self.output_size])
        self._built = True

    @using_scope
    def embed(self, x):
        return tf.nn.embedding_lookup(self.embedding_matrix, x)
