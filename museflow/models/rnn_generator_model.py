from cached_property import cached_property
import numpy as np
import tensorflow as tf

from ..components.component import using_scope
from ..components.rnn_decoder import RNNDecoder
from .model import Model


class RNNGeneratorModel(Model):

    def __init__(self, name='model', **config):
        super().__init__(name=name)
        self._config = config

        with self.use_scope():
            with tf.name_scope('inputs'):
                self.decoder_inputs = tf.placeholder(tf.int32, [None, None], 'decoder_inputs')
                self.decoder_targets = tf.placeholder(tf.int32, [None, None], 'decoder_targets')

                self._target_weights = tf.sign(self.decoder_targets)  # padding = 0
                self._target_length = tf.reduce_sum(self._target_weights, axis=1)

            self.decoder = RNNDecoder(**config['decoder'])

    @property
    def built(self):
        return self.decoder.built

    @cached_property
    @using_scope
    def train_outputs(self):
        return self.decoder.decode_train(self.decoder_inputs, self._target_length)

    @cached_property
    @using_scope
    def loss(self):
        with tf.name_scope('loss'):
            train_logits, _ = self.train_outputs
            batch_size = tf.shape(train_logits)[0]
            train_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.decoder_targets,
                logits=train_logits)
            return (tf.reduce_sum(train_xent * tf.to_float(self._target_weights)) /
                tf.to_float(batch_size))

    @cached_property
    @using_scope
    def train_summaries(self):
        return [tf.summary.scalar('train/loss', self.loss)]
