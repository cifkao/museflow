import argparse
import logging
import sys

import yaml

from museflow import scripts, models


def main():
    def print_help_and_exit(args):
        del args
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=print_help_and_exit)
    subparsers = parser.add_subparsers(title='commands')

    subparser = subparsers.add_parser('script')
    subsubparsers = subparser.add_subparsers(title='scripts')
    scripts.add_argparsers(subsubparsers)

    subparser = subparsers.add_parser('model')
    subparser.set_defaults(func=_run_model)
    subsubparsers = subparser.add_subparsers(title='models')
    _add_model_argparsers(subsubparsers)

    args = parser.parse_args()
    args.func(args)


def _add_model_argparsers(subparsers):
    for model_class in models.MODELS:
        subparser = subparsers.add_parser(model_class.__name__, description=model_class.__doc__)
        subparser.set_defaults(model_class=model_class)
        subparser.add_argument('--config', type=str, required=True,
                               help='path to the YAML configuration file')
        subparser.add_argument('--logdir', type=str, required=True, help='model directory')
        model_class.setup_argparser(subparser)

def _run_model(args):
    with open(args.config, 'rb') as f:
        config = yaml.load(f)
    model = args.model_class.from_args(args, config, logdir=args.logdir)
    model.run_action(args)
