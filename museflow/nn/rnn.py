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
