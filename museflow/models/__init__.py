import sys

from .rnn_generator import RNNGenerator
from .rnn_seq2seq import RNNSeq2Seq

MODELS = [RNNGenerator, RNNSeq2Seq]


def add_argparsers(subparsers):
    for model_class in MODELS:
        module = sys.modules[model_class.__module__]
        parser = subparsers.add_parser(model_class.__name__, description=model_class.__doc__)
        parser.set_defaults(func=module.main)
        parser.add_argument('--config', type=str, help='path to the YAML configuration file')
        parser.add_argument('--logdir', type=str, required=True, help='model directory')
        module.setup_argparser(parser)
