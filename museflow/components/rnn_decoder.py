import tensorflow as tf

from museflow.config import configurable
from .component import Component, using_scope


@configurable(['cell', 'output_projection', 'attention_wrapper'])
class RNNDecoder(Component):

    def __init__(self, vocabulary, embedding_layer, attention_mechanism=None, max_length=None,
                 name='decoder'):
        Component.__init__(self, name=name)

        self._vocabulary = vocabulary
        self._embeddings = embedding_layer
        self._attention_mechanism = attention_mechanism
        self._max_length = max_length

        with self.use_scope():
            self.cell = self._cfg.configure('cell', tf.nn.rnn_cell.GRUCell, dtype=tf.float32)
            if self._attention_mechanism:
                self.cell = self._cfg.configure('attention_wrapper',
                                                tf.contrib.seq2seq.AttentionWrapper,
                                                cell=self.cell,
                                                attention_mechanism=self._attention_mechanism,
                                                output_attention=False)
            self.cell.build(tf.TensorShape([None, self._embeddings.output_size]))

            self._output_projection = self._cfg.configure('output_projection', tf.layers.Dense,
                                                          units=len(vocabulary), use_bias=False,
                                                          name='output_projection')
            self._output_projection.build([None, self.cell.output_size])
        self._built = True

    @using_scope
    def decode_train(self, inputs, targets, initial_state=None):
        target_weights = tf.sign(targets, name='target_weights')
        embedded_inputs = self._embeddings.embed(inputs)

        with tf.name_scope('decode_train'):
            if initial_state is None:
                initial_state = self.cell.zero_state(batch_size=tf.shape(inputs)[0],
                                                     dtype=tf.float32)

            sequence_length = tf.reduce_sum(target_weights, axis=1)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=embedded_inputs,
                sequence_length=sequence_length,
                time_major=False)
            output = self._dynamic_decode(helper=helper,
                                          initial_state=initial_state)
            logits = output.rnn_output

        with tf.name_scope('loss'):
            batch_size = tf.shape(logits)[0]
            train_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits)
            loss = (tf.reduce_sum(train_xent * tf.to_float(target_weights)) /
                    tf.to_float(batch_size))

        return output, loss

    @using_scope
    def decode(self, initial_state=None, max_length=None, batch_size=None,
               softmax_temperature=1., random_seed=None, mode='greedy'):
        with tf.name_scope('decode_{}'.format(mode)):
            if batch_size is None:
                batch_size = tf.shape(initial_state)[0]
            if initial_state is None:
                initial_state = self.cell.zero_state(batch_size=batch_size,
                                                     dtype=tf.float32)

        return self._dynamic_decode(
            helper=self._make_helper(batch_size, softmax_temperature, random_seed, mode),
            initial_state=initial_state,
            max_length=max_length or self._max_length)

    def _make_helper(self, batch_size, softmax_temperature, random_seed, mode):
        helper_kwargs = {
            'embedding': self._embeddings.embedding_matrix,
            'start_tokens': tf.tile([self._vocabulary.start_id], [batch_size]),
            'end_token': self._vocabulary.end_id
        }

        if mode == 'greedy':
            return tf.contrib.seq2seq.GreedyEmbeddingHelper(**helper_kwargs)
        if mode == 'sample':
            helper_kwargs['softmax_temperature'] = softmax_temperature
            helper_kwargs['seed'] = random_seed
            return tf.contrib.seq2seq.SampleEmbeddingHelper(**helper_kwargs)

        raise ValueError('Unrecognized mode {!r}'.format(mode))

    def _dynamic_decode(self, helper, initial_state, max_length=None):
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=self._output_projection)
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_length)
        return output
