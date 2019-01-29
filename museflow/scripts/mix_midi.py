"""Mix multiple MIDI files.

This is done simply by combining the instrument tracks from all the files. Metadata such as tempo
and key signatures are taken from the first file.
"""

import pretty_midi


def setup_argparser(parser):
    parser.set_defaults(func=main)
    parser.add_argument('input_files', nargs='+', metavar='FILE')
    parser.add_argument('output_file', metavar='OUTPUTFILE')


def main(args):
    midi = pretty_midi.PrettyMIDI(args.input_files[0])

    for path in args.input_files[1:]:
        midi2 = pretty_midi.PrettyMIDI(path)
        midi.instruments.extend(midi2.instruments)

    midi.write(args.output_file)
