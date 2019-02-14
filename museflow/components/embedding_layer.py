import tensorflow as tf

from .component import Component, using_scope


class EmbeddingLayer(Component):

    def __init__(self, input_size, output_size, name='embedding'):
        Component.__init__(self, name=name)

        self.input_size = input_size
        self.output_size = output_size

        with self.use_scope():
            self.embedding_matrix = tf.get_variable(
                'embedding_matrix', shape=[self.input_size, self.output_size])
        self._built = True

    @using_scope
    def embed(self, x):
        return tf.nn.embedding_lookup(self.embedding_matrix, x)

    def __call__(self, inputs):
        return self.embed(inputs)
