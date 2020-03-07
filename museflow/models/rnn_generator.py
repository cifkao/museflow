import argparse
import os
import pickle

from confugue import Configuration, configurable
import tensorflow as tf

from museflow import logger
from museflow.components import RNNDecoder, EmbeddingLayer
from museflow.model_utils import (DatasetManager, create_train_op, prepare_train_and_val_data,
                                  set_random_seed)
from museflow.trainer import BasicTrainer


@configurable
class RNNGenerator:

    def __init__(self, train_mode, vocabulary, sampling_seed=None):
        self._train_mode = train_mode
        self._is_training = tf.placeholder_with_default(False, [], name='is_training')

        self.dataset_manager = DatasetManager(
            output_types=(tf.int32, tf.int32),
            output_shapes=([None, None], [None, None]))

        embeddings = self._cfg['embedding_layer'].configure(EmbeddingLayer,
                                                            input_size=len(vocabulary))
        decoder = self._cfg['decoder'].configure(RNNDecoder,
                                                 vocabulary=vocabulary,
                                                 embedding_layer=embeddings,
                                                 training=self._is_training)

        # Build the training version of the decoder and the training ops
        self.training_ops = None
        if train_mode:
            inputs, targets = self.dataset_manager.get_batch()
            _, self.loss = decoder.decode_train(inputs, targets)
            self.training_ops = self._make_train_ops()

        # Build the sampling version of the decoder
        self.sample_batch_size = tf.placeholder(tf.int32, [], name='sample_batch_size')
        self.softmax_temperature = tf.placeholder(tf.float32, [], name='softmax_temperature')
        self.sample_outputs, _ = decoder.decode(mode='sample',
                                                batch_size=self.sample_batch_size,
                                                softmax_temperature=self.softmax_temperature,
                                                random_seed=sampling_seed)

    def _make_train_ops(self):
        train_op = self._cfg['training'].configure(create_train_op, loss=self.loss)
        init_op = tf.global_variables_initializer()

        tf.summary.scalar('train/loss', self.loss)
        train_summary_op = tf.summary.merge_all()

        return BasicTrainer.TrainingOps(loss=self.loss,
                                        train_op=train_op,
                                        init_op=init_op,
                                        summary_op=train_summary_op,
                                        training_placeholder=self._is_training)

    def sample(self, session, batch_size, softmax_temperature=1.):
        _, sample_ids = session.run(
            self.sample_outputs,
            {self.sample_batch_size: batch_size, self.softmax_temperature: softmax_temperature}
        )
        return sample_ids


def setup_argparser(parser):
    subparsers = parser.add_subparsers(title='action', dest='action')
    subparsers.add_parser('train')
    subparser = subparsers.add_parser('sample')
    subparser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE')
    subparser.add_argument('--checkpoint', default=None, type=str)
    subparser.add_argument('--batch-size', default=1, type=int)
    subparser.add_argument('--softmax-temperature', default=1., type=float)
    subparser.add_argument('--seed', default=None, type=int, dest='sampling_seed')


def main(args):
    config_file = args.config or os.path.join(args.logdir, 'model.yaml')
    with open(config_file, 'rb') as f:
        config = Configuration.from_yaml(f)
    logger.debug(config)

    model, trainer, encoding = config.configure(
        _init, logdir=args.logdir, train_mode=(args.action == 'train'),
        sampling_seed=getattr(args, 'seed', None))

    if args.action == 'train':
        trainer.train()
    elif args.action == 'sample':
        trainer.load_variables(checkpoint_file=args.checkpoint)
        output_ids = model.sample(session=trainer.session,
                                  batch_size=args.batch_size,
                                  softmax_temperature=args.softmax_temperature)
        output = [encoding.decode(seq) for seq in output_ids]
        pickle.dump(output, args.output_file)


@configurable
def _init(logdir, train_mode, *, sampling_seed=None, _cfg):
    set_random_seed(_cfg.get('random_seed', None))

    encoding = _cfg['encoding'].configure()
    model = _cfg['model'].configure(RNNGenerator,
                                    train_mode=train_mode,
                                    vocabulary=encoding.vocabulary,
                                    sampling_seed=sampling_seed)
    trainer = _cfg['trainer'].configure(BasicTrainer,
                                        dataset_manager=model.dataset_manager,
                                        training_ops=model.training_ops,
                                        logdir=logdir,
                                        write_summaries=train_mode)

    if train_mode:
        # Configure the dataset manager with the training and validation data.
        _cfg['data_prep'].configure(
            prepare_train_and_val_data,
            dataset_manager=model.dataset_manager,
            train_generator=_make_data_generator(encoding, _cfg.get('train_data')),
            val_generator=_make_data_generator(encoding, _cfg.get('val_data')),
            output_types=(tf.int32, tf.int32),
            output_shapes=([None], [None]))

    return model, trainer, encoding


def _make_data_generator(encoding, fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    def generator():
        for _, example in data:
            encoded = encoding.encode(example, add_start=True, add_end=True)
            yield encoded[:-1], encoded[1:]

    return generator
