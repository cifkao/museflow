import tensorflow as tf

from museflow.config import configurable
from museflow.nn.rnn import DropoutWrapper
from .component import Component, using_scope


@configurable(['forward_cell', 'backward_cell', 'dropout'])
class RNNLayer(Component):

    def __init__(self, training=None, name='rnn'):
        Component.__init__(self, name=name)

        with self.use_scope():
            fw_cell = self._cfg.configure('forward_cell', tf.nn.rnn_cell.GRUCell, dtype=tf.float32)
            fw_cell_dropout = self._cfg.maybe_configure('dropout', DropoutWrapper,
                                                        cell=fw_cell, training=training)
            self._fw_cell = fw_cell_dropout or fw_cell

            self._bw_cell = self._cfg.maybe_configure('backward_cell', tf.nn.rnn_cell.GRUCell,
                                                      dtype=tf.float32)
            if self._bw_cell:
                bw_cell_dropout = self._cfg.maybe_configure('dropout', DropoutWrapper,
                                                            cell=self._bw_cell, training=training)
                self._bw_cell = bw_cell_dropout or self._bw_cell

    @using_scope
    def apply(self, inputs):
        if not self._bw_cell:
            return tf.nn.dynamic_rnn(self._fw_cell, inputs, dtype=tf.float32)

        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            self._fw_cell, self._bw_cell, inputs, dtype=tf.float32)
        outputs = tf.concat(outputs, -1)
        final_states = tf.concat(final_states, -1)

        return outputs, final_states

    def __call__(self, inputs):
        return self.apply(inputs)
