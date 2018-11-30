import argparse
import itertools
import pickle
import sys

import pretty_midi
import numpy as np
import tensorflow as tf
import yaml

from museflow.config import Configurable
from museflow.components import RNNDecoder
from museflow.encodings import PerformanceEncoding
from museflow.training import create_train_op, BasicTrainer


class RNNGenerator(Configurable):
    _subconfigs = ['encoding', 'trainer', 'decoder', 'rnn_cell', 'optimizer']

    def __init__(self, logdir, train_mode, config=None, **kwargs):
        Configurable.__init__(self, config)
        self._train_mode = train_mode
        self._logdir = logdir
        self._args = kwargs

        self._session = tf.Session()
        self._encoding = self._configure('encoding')
        self._trainer = self._configure('trainer', BasicTrainer,
                                        logdir=self._logdir, session=self._session)

        self._decoder = self._configure(
            'decoder', RNNDecoder,
            cell=self._configure('rnn_cell', tf.nn.rnn_cell.GRUCell, dtype=tf.float32),
            vocabulary=self._encoding.vocabulary)

        # Build the training and sampling versions of the decoder
        if train_mode:
            inputs, targets = self._prepare_data()
            _, self._loss = self._decoder.decode_train(inputs, targets)
        self._sample_batch_size = tf.placeholder(tf.int32, [])
        self._softmax_temperature = tf.placeholder(tf.float32, [])
        self._sample_outputs = self._decoder.decode(mode='sample',
                                                    batch_size=self._sample_batch_size,
                                                    softmax_temperature=self._softmax_temperature)

        if train_mode:
            self._init_op, self._train_op, self._train_summary_op = self._make_train_ops()

    def _prepare_data(self):
        with tf.name_scope('data'):
            with tf.name_scope('train'):
                train_dataset = tf.data.Dataset.from_generator(
                    self._make_data_generator(self._args['train_data']),  (tf.int32, tf.int32))
                train_dataset = train_dataset.shuffle(10000, reshuffle_each_iteration=True).repeat()
                train_dataset = train_dataset.padded_batch(
                    self._args['train_batch_size'], ([None], [None]))

            with tf.name_scope('val'):
                val_dataset = tf.data.Dataset.from_generator(
                    self._make_data_generator(self._args['val_data']), (tf.int32, tf.int32))
                val_dataset = val_dataset.padded_batch(
                    self._args['val_batch_size'], ([None], [None]))

            return self._trainer.prepare_data(train_dataset, val_dataset)

    def _make_data_generator(self, fname):
        def generator():
            with open(fname, 'rb') as f:
                data = pickle.load(f)

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
        subparser.add_argument('--batch_size', default=1, type=int)
        subparser.add_argument('--softmax_temperature', default=1., type=float)

    def run_action(self, args):
        if args.action == 'train':
            self.train()
        elif args.action == 'sample':
            self.load(checkpoint_file=args.checkpoint)
            output = self.sample(batch_size=args.batch_size,
                                 softmax_temperature=args.softmax_temperature)
            print(output)
