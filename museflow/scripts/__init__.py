import importlib


def add_argparsers(subparsers):
    for module_name in ['chop_midi', 'notes2midi', 'mix_midi']:
        module = importlib.import_module('.' + module_name, package=__name__)
        parser = subparsers.add_parser(module_name, description=module.__doc__)
        module.setup_argparser(parser)
