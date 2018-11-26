import argparse
import sys

from . import scripts


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands')

    subparser = subparsers.add_parser('script')
    script_subparsers = subparser.add_subparsers(title='scripts')
    scripts.add_parsers(script_subparsers)

    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
        sys.exit(1)
