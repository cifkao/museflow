"""Data processing utilities."""

import pickle

from ..data import chop_midi


def add_parser(parsers):
    parser = parsers.add_parser('data', description=__doc__)
    parser.set_defaults(func=main)
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    subparser = subparsers.add_parser(
        'chop_midi',
        help='Chop midi files into segments containing a given number of bars')
    subparser.add_argument('fnames', type=str, nargs='+', metavar='FILE')
    subparser.add_argument('ofname', type=str, metavar='OUTPUTFILE')
    subparser.add_argument('-i', '--instrument-re', type=str, default='.*')
    subparser.add_argument('-b', '--bars-per-segment', type=int, default=8)
    subparser.add_argument('-n', '--min-notes-per-segment', type=int, default=2)
    subparser.add_argument('--include-segment-id', action='store_true')


def main(args):
    if args.subcommand == 'chop_midi':
        output = list(chop_midi(midis=args.fnames,
                                instrument_re=args.instrument_re,
                                bars_per_segment=args.bars_per_segment,
                                min_notes_per_segment=args.min_notes_per_segment,
                                include_segment_id=args.include_segment_id))
        with open(args.ofname, 'wb') as f:
            pickle.dump(output, f)
