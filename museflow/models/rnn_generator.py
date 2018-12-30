import argparse
import itertools
import pickle
import sys

import pretty_midi
import numpy as np
import tensorflow as tf
import yaml

from museflow.config import Configurable
from museflow.components import RNNDecoder, EmbeddingLayer
from museflow.encodings import PerformanceEncoding
from museflow.model_utils import DatasetManager, create_train_op, prepare_train_and_val_data
from museflow.training import BasicTrainer


class RNNGenerator(Configurable):
    _subconfigs = ['data_prep', 'encoding', 'embedding_layer', 'decoder', 'trainer', 'optimizer']

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
                                train_generator=self._make_data_generator(self._args['train_data']),
                                val_generator=self._make_data_generator(self._args['val_data']),
                                output_types=(tf.int32, tf.int32),
                                output_shapes=([None], [None]))

        vocabulary = self._encoding.vocabulary
        embeddings = self._configure('embedding_layer', EmbeddingLayer, input_size=len(vocabulary))
        self._decoder = self._configure('decoder', RNNDecoder,
                                        vocabulary=vocabulary,
                                        embedding_layer=embeddings)

        # Build the training version of the decoder and the training ops
        if train_mode:
            inputs, targets = self._dataset_manager.get_batch()
            _, self._loss = self._decoder.decode_train(inputs, targets)
            self._init_op, self._train_op, self._train_summary_op = self._make_train_ops()

        # Build the sampling version of the decoder
        self._sample_batch_size = tf.placeholder(tf.int32, [], name='sample_batch_size')
        self._softmax_temperature = tf.placeholder(tf.float32, [], name='softmax_temperature')
        self._sample_outputs = self._decoder.decode(mode='sample',
                                                    batch_size=self._sample_batch_size,
                                                    softmax_temperature=self._softmax_temperature)

        self._session = tf.Session()
        self._trainer = self._configure('trainer', BasicTrainer,
                                        dataset_manager=self._dataset_manager,
                                        logdir=self._logdir, session=self._session)

    def _make_data_generator(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        def generator():
            for _, example in data:
                encoded = self._encoding.encode(example, add_start=True, add_end=True)
                yield encoded[:-1], encoded[1:]

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

    def sample(self, batch_size, softmax_temperature=1.):
        _, sample_ids = self._session.run(
            self._sample_outputs,
            {self._sample_batch_size: batch_size, self._softmax_temperature: softmax_temperature}
        )
        return [self._encoding.decode(seq) for seq in sample_ids]

    @classmethod
    def from_args(cls, args, config, logdir):
        return cls.from_config(
            config, logdir=logdir, train_mode=(args.action == 'train'))

    @classmethod
    def setup_argparser(cls, parser):
        subparsers = parser.add_subparsers(title='action', dest='action')
        subparsers.add_parser('train')
        subparser = subparsers.add_parser('sample')
        subparser.add_argument('--checkpoint', default=None, type=str)
        subparser.add_argument('--batch-size', default=1, type=int)
        subparser.add_argument('--softmax-temperature', default=1., type=float)

    def run_action(self, args):
        if args.action == 'train':
            self.train()
        elif args.action == 'sample':
            self.load(checkpoint_file=args.checkpoint)
            output = self.sample(batch_size=args.batch_size,
                                 softmax_temperature=args.softmax_temperature)
            print(output)
