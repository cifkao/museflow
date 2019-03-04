import argparse
import logging
import sys

import coloredlogs

from museflow import scripts, models


def main():
    def print_help_and_exit(args):
        del args
        parser.print_help()
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=print_help_and_exit)
    parser.add_argument('--color', action='store_true', default=None,
                        help='force colored logs')
    subparsers = parser.add_subparsers(title='commands')

    subparser = subparsers.add_parser('script')
    subsubparsers = subparser.add_subparsers(title='scripts')
    scripts.add_argparsers(subsubparsers)

    subparser = subparsers.add_parser('model')
    subsubparsers = subparser.add_subparsers(title='models')
    models.add_argparsers(subsubparsers)

    args = parser.parse_args()

    coloredlogs.install(level='DEBUG', logger=logging.getLogger(), isatty=args.color)
    logging.captureWarnings(True)

    args.func(args)
