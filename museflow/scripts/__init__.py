import importlib


def add_parsers(subparsers):
    for module_name in ['chop_midi']:
        module = importlib.import_module('.' + module_name, package=__name__)
        parser = subparsers.add_parser(module_name, description=module.__doc__)
        module.setup_parser(parser)
