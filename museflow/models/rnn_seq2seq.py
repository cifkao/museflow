import argparse
import fileinput
import itertools
import pickle
import sys

import pretty_midi
import numpy as np
import tensorflow as tf
import yaml

from museflow.config import Configurable
from museflow.components import EmbeddingLayer, RNNEncoder, RNNDecoder
from museflow.encodings import PerformanceEncoding
from museflow.model_utils import (DatasetManager, create_train_op, prepare_train_and_val_data,
                                  make_simple_dataset)
from museflow.training import BasicTrainer


class RNNSeq2Seq(Configurable):
    _subconfigs = ['data_prep', 'encoding', 'trainer', 'embedding_layer', 'encoder',
                   'state_projection', 'decoder', 'optimizer']

    def __init__(self, logdir, train_mode, config=None, **kwargs):
        Configurable.__init__(self, config)
        self._train_mode = train_mode
        self._logdir = logdir
        self._args = kwargs

        self._encoding = self._configure('encoding')

        with tf.name_scope('data'):
            self._dataset_manager = DatasetManager()
            if train_mode:
                # Configure the dataset manager with the training and validation data.
                self._configure('data_prep', prepare_train_and_val_data,
                                dataset_manager=self._dataset_manager,
                                train_generator=self._load_data(self._args['train_data']['src'],
                                                                self._args['train_data']['tgt']),
                                val_generator=self._load_data(self._args['val_data']['src'],
                                                              self._args['val_data']['tgt']),
                                output_types=(tf.int32, tf.int32, tf.int32),
                                output_shapes=([None], [None], [None]))

        vocabulary = self._encoding.vocabulary
        embeddings = self._configure('embedding_layer', EmbeddingLayer, input_size=len(vocabulary))
        encoder = self._configure('encoder', RNNEncoder)
        self._decoder = self._configure('decoder', RNNDecoder,
                                        vocabulary=vocabulary,
                                        embedding_layer=embeddings)

        inputs, decoder_inputs, decoder_targets = self._dataset_manager.get_batch()
        _, encoder_final_state = encoder.encode(embeddings.embed(inputs))
        state_projection = self._configure('state_projection', tf.layers.Dense,
                                           units=self._decoder.cell.state_size)
        decoder_initial_state = state_projection(encoder_final_state)

        # Build the training version of the decoder and the training ops
        if train_mode:
            _, self._loss = self._decoder.decode_train(decoder_inputs, decoder_targets,
                                                       initial_state=decoder_initial_state)
            self._init_op, self._train_op, self._train_summary_op = self._make_train_ops()

        # Build the sampling and greedy version of the decoder
        self._sample_batch_size = tf.placeholder(tf.int32, [])
        self._softmax_temperature = tf.placeholder(tf.float32, [])
        self._sample_outputs = self._decoder.decode(mode='sample',
                                                    batch_size=self._sample_batch_size,
                                                    softmax_temperature=self._softmax_temperature,
                                                    initial_state=decoder_initial_state)
        self._greedy_outputs = self._decoder.decode(mode='greedy',
                                                    batch_size=self._sample_batch_size,
                                                    initial_state=decoder_initial_state)

        self._session = tf.Session()
        self._trainer = self._configure('trainer', BasicTrainer,
                                        dataset_manager=self._dataset_manager,
                                        logdir=self._logdir, session=self._session)

    def _load_data(self, src_fname, tgt_fname):
        with open(src_fname, 'rb') as f:
            src_data = pickle.load(f)
        with open(tgt_fname, 'rb') as f:
            tgt_data = pickle.load(f)
        return self._make_data_generator(src_data, tgt_data)

    def _make_data_generator(self, src_data, tgt_data=None):
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
                tgt_notes = tgt_data[segment_id] if tgt_data else []

                src_encoded = self._encoding.encode(src_notes, add_start=False, add_end=False)
                tgt_encoded = self._encoding.encode(tgt_notes, add_start=True, add_end=True)
                yield src_encoded, tgt_encoded[:-1], tgt_encoded[1:]

        return generator

    def _make_train_ops(self):
        train_op = create_train_op(
            self._configure('optimizer', tf.train.AdamOptimizer),
            self._loss,
            self._decoder.trainable_variables)
        init_op = tf.global_variables_initializer()

        tf.summary.scalar('train/loss', self._loss)
        train_summary_op = tf.summary.merge_all(scope='train')

        return init_op, train_op, train_summary_op

    def train(self):
        self._trainer.train(train_op=self._train_op, loss=self._loss, init_op=self._init_op,
                            train_summary_op=self._train_summary_op)

    def load(self, checkpoint_name='best', checkpoint_file=None):
        self._trainer.load_variables(checkpoint_name, checkpoint_file)

    def run(self, data, batch_size, sample=False, softmax_temperature=1.):
        dataset = make_simple_dataset(
            self._make_data_generator(data),
            output_types=(tf.int32, tf.int32, tf.int32),
            output_shapes=([None], [None], [None]),
            batch_size=batch_size)
        outputs_tensor = self._sample_outputs if sample else self._greedy_outputs

        return self._dataset_manager.run_over_dataset(
            self._session, outputs_tensor, dataset,
            {self._softmax_temperature: softmax_temperature})

    @classmethod
    def from_args(cls, args, config, logdir):
        return cls.from_config(
            config, logdir=logdir, train_mode=(args.action == 'train'))

    @classmethod
    def setup_argparser(cls, parser):
        subparsers = parser.add_subparsers(title='action', dest='action')
        subparsers.add_parser('train')
        subparser = subparsers.add_parser('run')
        subparser.add_argument('input_file', type=argparse.FileType('rb'), metavar='INPUTFILE')
        subparser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE')
        subparser.add_argument('--checkpoint', default=None, type=str)
        subparser.add_argument('--batch-size', default=None, type=int)
        subparser.add_argument('--sample', action='store_true')
        subparser.add_argument('--softmax-temperature', default=1., type=float)

    def run_action(self, args):
        if args.action == 'train':
            self.train()
        elif args.action == 'run':
            data = pickle.load(args.input_file)
            output = self.run(data, args.batch_size, args.sample, args.softmax_temperature)
            pickle.dump(output, args.output_file)