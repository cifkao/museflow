from confugue import configurable
import tensorflow as tf

from museflow.nn.rnn import DropoutWrapper
from .component import Component, using_scope


@configurable(params=['output_states', 'name'])
class RNNLayer(Component):

    def __init__(self, training=None, forward_cell=None, backward_cell=None,
                 output_states='all', name='rnn'):
        Component.__init__(self, name=name)

        if output_states not in ['all', 'output', 'final']:
            raise ValueError(f"Invalid value for output_states: '{output_states}'; "
                             "expected 'output', 'final' or 'all'")
        self._output_states = output_states
        self._training = training

        with self.use_scope():
            if forward_cell:
                fw_cell = forward_cell
            else:
                fw_cell = self._cfg['forward_cell'].configure(tf.nn.rnn_cell.GRUCell,
                                                              dtype=tf.float32)
            fw_cell_dropout = self._cfg['dropout'].maybe_configure(
                DropoutWrapper, cell=fw_cell, training=self._training)
            self._fw_cell = fw_cell_dropout or fw_cell

            if backward_cell:
                self._bw_cell = backward_cell
            else:
                self._bw_cell = self._cfg['backward_cell'].maybe_configure(tf.nn.rnn_cell.GRUCell,
                                                                           dtype=tf.float32)
            if self._bw_cell:
                bw_cell_dropout = self._cfg['dropout'].maybe_configure(
                    DropoutWrapper, cell=self._bw_cell, training=self._training)
                self._bw_cell = bw_cell_dropout or self._bw_cell

            self._final_dropout = self._cfg['final_state_dropout'].maybe_configure(
                tf.layers.Dropout)

    @using_scope
    def apply(self, inputs):
        if not self._bw_cell:
            outputs, final_states = tf.nn.dynamic_rnn(self._fw_cell, inputs, dtype=tf.float32)
        else:
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                self._fw_cell, self._bw_cell, inputs, dtype=tf.float32)
            outputs = tf.concat(outputs, -1)
            final_states = tf.concat(final_states, -1)

        if self._final_dropout:
            final_states = self._final_dropout(final_states, training=self._training)

        if self._output_states == 'output':
            return outputs
        elif self._output_states == 'final':
            return final_states
        else:
            return outputs, final_states

    def __call__(self, inputs):
        return self.apply(inputs)
