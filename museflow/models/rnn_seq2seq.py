import argparse
import os
import pickle

import tensorflow as tf
import yaml

from museflow import logger
from museflow.components import EmbeddingLayer, RNNEncoder, RNNDecoder
from museflow.config import configurable, configure
from museflow.model_utils import (DatasetManager, create_train_op, prepare_train_and_val_data,
                                  make_simple_dataset, set_random_seed)
from museflow.trainer import BasicTrainer


@configurable(['embedding_layer', 'encoder', 'state_projection',
               'decoder', 'attention_mechanism', 'training'])
class RNNSeq2Seq:

    def __init__(self, train_mode, vocabulary, sampling_seed=None):
        self._train_mode = train_mode
        self._is_training = tf.placeholder_with_default(False, [], name='is_training')

        self.dataset_manager = DatasetManager(
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([None, None], [None, None], [None, None]))

        inputs, decoder_inputs, decoder_targets = self.dataset_manager.get_batch()
        batch_size = tf.shape(inputs)[0]

        embeddings = self._cfg.configure('embedding_layer', EmbeddingLayer,
                                         input_size=len(vocabulary))
        encoder = self._cfg.configure('encoder', RNNEncoder,
                                      training=self._is_training)
        encoder_states, encoder_final_state = encoder.encode(embeddings.embed(inputs))

        with tf.variable_scope('attention'):
            attention = self._cfg.maybe_configure('attention_mechanism', memory=encoder_states)
        decoder = self._cfg.configure('decoder', RNNDecoder,
                                      vocabulary=vocabulary,
                                      embedding_layer=embeddings,
                                      attention_mechanism=attention,
                                      training=self._is_training)

        state_projection = self._cfg.configure('state_projection', tf.layers.Dense,
                                               units=decoder.initial_state_size,
                                               name='state_projection')
        decoder_initial_state = state_projection(encoder_final_state)

        # Build the training version of the decoder and the training ops
        self.training_ops = None
        if train_mode:
            _, self.loss = decoder.decode_train(decoder_inputs, decoder_targets,
                                                initial_state=decoder_initial_state)
            self.training_ops = self._make_train_ops()

        # Build the sampling and greedy version of the decoder
        self.softmax_temperature = tf.placeholder(tf.float32, [], name='softmax_temperature')
        self.sample_outputs = decoder.decode(mode='sample',
                                             softmax_temperature=self.softmax_temperature,
                                             initial_state=decoder_initial_state,
                                             batch_size=batch_size,
                                             random_seed=sampling_seed)
        self.greedy_outputs = decoder.decode(mode='greedy',
                                             initial_state=decoder_initial_state,
                                             batch_size=batch_size)

    def _make_train_ops(self):
        train_op = self._cfg.configure('training', create_train_op, loss=self.loss)
        init_op = tf.global_variables_initializer()

        tf.summary.scalar('train/loss', self.loss)
        train_summary_op = tf.summary.merge_all()

        return BasicTrainer.TrainingOps(loss=self.loss,
                                        train_op=train_op,
                                        init_op=init_op,
                                        summary_op=train_summary_op,
                                        training_placeholder=self._is_training)

    def run(self, session, dataset, sample=False, softmax_temperature=1.):
        _, output_ids_tensor = self.sample_outputs if sample else self.greedy_outputs

        return self.dataset_manager.run_over_dataset(
            session, output_ids_tensor, dataset,
            feed_dict={self.softmax_temperature: softmax_temperature},
            concat_batches=True)


def setup_argparser(parser):
    subparsers = parser.add_subparsers(title='action', dest='action')
    subparsers.add_parser('train')
    subparser = subparsers.add_parser('run')
    subparser.add_argument('input_file', type=argparse.FileType('rb'), metavar='INPUTFILE')
    subparser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE')
    subparser.add_argument('--checkpoint', default=None, type=str)
    subparser.add_argument('--batch-size', default=32, type=int)
    subparser.add_argument('--sample', action='store_true')
    subparser.add_argument('--softmax-temperature', default=1., type=float)
    subparser.add_argument('--seed', default=None, type=int, dest='sampling_seed')


def main(args):
    config_file = args.config or os.path.join(args.logdir, 'model.yaml')
    with open(config_file, 'rb') as f:
        config_dict = yaml.load(f)
    logger.debug(config_dict)

    model, trainer, encoding = configure(
        _init, config_dict, logdir=args.logdir,
        train_mode=(args.action == 'train'),
        sampling_seed=getattr(args, 'seed', None))

    if args.action == 'train':
        trainer.train()
    elif args.action == 'run':
        trainer.load_variables(checkpoint_file=args.checkpoint)
        data = pickle.load(args.input_file)
        dataset = make_simple_dataset(
            _make_data_generator(encoding, data),
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([None], [None], [None]),
            batch_size=args.batch_size)

        output_ids = model.run(trainer.session, dataset, args.sample, args.softmax_temperature)
        output = [encoding.decode(seq) for seq in output_ids]
        pickle.dump(output, args.output_file)


@configurable(['encoding', 'model', 'data_prep', 'train_data', 'val_data', 'trainer'])
def _init(cfg, logdir, train_mode, **kwargs):
    set_random_seed(kwargs.get('random_seed'))

    encoding = cfg.configure('encoding')
    model = cfg.configure('model', RNNSeq2Seq,
                          train_mode=train_mode,
                          vocabulary=encoding.vocabulary,
                          sampling_seed=kwargs.get('sampling_seed'))
    trainer = cfg.configure('trainer', BasicTrainer,
                            dataset_manager=model.dataset_manager,
                            training_ops=model.training_ops,
                            logdir=logdir)

    if train_mode:
        # Configure the dataset manager with the training and validation data.
        cfg.configure(
            'data_prep', prepare_train_and_val_data,
            dataset_manager=model.dataset_manager,
            train_generator=cfg.configure('train_data', _load_data, encoding=encoding),
            val_generator=cfg.configure('val_data', _load_data, encoding=encoding),
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([None], [None], [None]))

    return model, trainer, encoding


def _load_data(encoding, src, tgt):
    with open(src, 'rb') as f:
        src_data = pickle.load(f)
    with open(tgt, 'rb') as f:
        tgt_data = pickle.load(f)
    return _make_data_generator(encoding, src_data, tgt_data)


def _make_data_generator(encoding, src_data, tgt_data=None):
    if tgt_data:
        # tgt_data is a list of tuples (segment_id, notes)
        tgt_data = dict(tgt_data)

    def generator():
        for src_example in src_data:
            if isinstance(src_example, tuple):
                segment_id, src_notes = src_example
            else:
                src_notes = src_example

            # Get the target segment corresponding to this source segment.
            # If targets are not available, feed an empty sequence.
            try:
                tgt_notes = tgt_data[segment_id] if tgt_data else []
            except KeyError as e:
                logger.warning(f'KeyError: {e}')
                continue

            src_encoded = encoding.encode(src_notes, add_start=False, add_end=False)
            tgt_encoded = encoding.encode(tgt_notes, add_start=True, add_end=True)
            yield src_encoded, tgt_encoded[:-1], tgt_encoded[1:]

    return generator
