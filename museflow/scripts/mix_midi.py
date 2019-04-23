"""Mix multiple MIDI files.

This is done simply by combining the instrument tracks from all the files. Metadata such as tempo
and key signatures are taken from the first file.
"""

import pretty_midi

from .chop_midi import normalize_tempo


def setup_argparser(parser):
    parser.set_defaults(func=main)
    parser.add_argument('input_files', nargs='+', metavar='FILE')
    parser.add_argument('output_file', metavar='OUTPUTFILE')
    parser.add_argument('--warp', action='store_true')


def main(args):
    midi = pretty_midi.PrettyMIDI(args.input_files[0])
    if args.warp:
        tempo_change_times, tempi = midi.get_tempo_changes()
        tempo_change_beats = [midi.time_to_tick(t) / midi.resolution for t in tempo_change_times]

    for path in args.input_files[1:]:
        midi2 = pretty_midi.PrettyMIDI(path)
        if args.warp:
            # Normalize the tempo so that time in seconds is equal to time in beats.
            normalize_tempo(midi2, 60)

            # Now apply the tempi of the first file.
            end_time = midi2.get_end_time()
            if end_time > tempo_change_beats[-1]:
                # Extrapolate the last tempo until the end of this file.
                new_end_time = (tempo_change_times[-1]
                                + (end_time - tempo_change_beats[-1]) * 60 / tempi[-1])
            else:
                new_end_time = end_time
            midi2.adjust_times([*tempo_change_beats, end_time],
                               [*tempo_change_times, new_end_time])

        midi.instruments.extend(midi2.instruments)

    midi.write(args.output_file)
