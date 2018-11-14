import argparse
import sys

from . import commands

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands')
    for module in [commands.data]:
        module.add_parser(subparsers)
    args = parser.parse_args()

    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
        sys.exit(1)
