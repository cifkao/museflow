import tensorflow as tf


class DropoutWrapper(tf.nn.rnn_cell.DropoutWrapper):  # pylint: disable=abstract-method
    """A version of `tf.nn.rnn_cell.DropoutWrapper` that disables dropout during inference."""

    def __init__(self, cell, training, **kwargs):
        """Initialize the wrapper.

        Args:
            cell: An `RNNCell`.
            training: A `tf.bool` tensor indicating whether we are in training mode.
            **kwargs: Any other arguments to `tf.nn.rnn_cell.DropoutWrapper`.
        """
        for key in ['input_keep_prob', 'output_keep_prob', 'state_keep_prob']:
            if key in kwargs:
                kwargs[key] = tf.cond(training,
                                      lambda key=key: tf.convert_to_tensor(kwargs[key]),
                                      lambda: tf.constant(1.))

        super().__init__(cell, **{'dtype': tf.float32, **kwargs})


class InputWrapper(tf.nn.rnn_cell.RNNCell):
    """A wrapper for passing additional input to an RNN cell."""

    def __init__(self, cell, input_fn):
        """Initialize the wrapper.

        Args:
            cell: An `RNNCell`.
            input_fn: A function expecting a scalar tensor argument `batch_size` and returning
                a tensor of shape `[batch_size, input_size]` to concatenate with the RNN cell input.
        """
        super().__init__()
        self._cell = cell
        self._dtype = self._cell.dtype
        self._input_fn = input_fn

    @property
    def wrapped_cell(self):
        return self._cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        return self._cell.compute_output_shape(input_shape)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        batch_size = tf.shape(state)[0]
        inputs = tf.concat([inputs, self._input_fn(batch_size)], axis=-1)
        return self._cell(inputs, state, scope=scope)
