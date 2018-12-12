import tensorflow as tf

from museflow.config import Configurable
from .component import Component, using_scope


class RNNEncoder(Component, Configurable):
    _subconfigs = ['forward_cell', 'backward_cell']

    def __init__(self, name='encoder', config=None):
        Component.__init__(self, name=name)
        Configurable.__init__(self, config)

        with self.use_scope():
            self._fw_cell = self._configure('forward_cell', tf.nn.rnn_cell.GRUCell,
                                            dtype=tf.float32)
            self._bw_cell = self._configure('backward_cell', tf.nn.rnn_cell.GRUCell,
                                            dtype=tf.float32) if 'backward_cell' in config else None

    @using_scope
    def encode(self, inputs):
        if self._bw_cell is None:
            return tf.nn.dynamic_rnn(self._fw_cell, inputs, dtype=tf.float32)

        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            self._fw_cell, self._bw_cell, inputs, dtype=tf.float32)
        outputs = tf.concat(outputs, -1)
        final_states = tf.concat(final_states, -1)

        return outputs, final_states
