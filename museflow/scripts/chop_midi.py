"""Chop midi files into segments containing a given number of bars."""

import argparse
import math
import pickle
import re
import sys

import pretty_midi


def setup_parser(parser):
    parser.set_defaults(func=main)
    parser.add_argument('fnames', type=str, nargs='+', metavar='FILE')
    parser.add_argument('ofname', type=str, metavar='OUTPUTFILE')
    parser.add_argument('-i', '--instrument-re', type=str, default='.*')
    parser.add_argument('-b', '--bars-per-segment', type=int, default=8)
    parser.add_argument('-n', '--min-notes-per-segment', type=int, default=2)
    parser.add_argument('-t', '--force-tempo', type=float, default=None)
    parser.add_argument('--skip-bars', type=int, default=0)
    parser.add_argument('--include-segment-id', action='store_true')


def chop_midi(midis, instrument_re, bars_per_segment, min_notes_per_segment=2,
              include_segment_id=False, force_tempo=None, skip_bars=0):
    for file_index, midi in enumerate(midis):
        file_id = file_index
        if not isinstance(midi, pretty_midi.PrettyMIDI):
            file_id = str(midi)
            midi = pretty_midi.PrettyMIDI(midi)

        if force_tempo is not None:
            normalize_tempo(midi, force_tempo)

        instruments = [i for i in midi.instruments if re.search(instrument_re, i.name)]
        if len(instruments) < 1:
            print(
                'Regex {!r} does not match any track in file {!r}; skipping file'
                .format(instrument_re, file_id), file=sys.stderr)
            continue
        all_notes = [n for i in instruments for n in i.notes]

        downbeats = midi.get_downbeats()[skip_bars:]
        bps = bars_per_segment
        for i in range(0, len(downbeats), bps):
            start = downbeats[i]
            end = downbeats[i + bps] if i + bps < len(downbeats) else midi.get_end_time()
            if math.isclose(start, end):
                continue

            # Find notes that overlap with this segment and clip them.
            notes = [
                pretty_midi.Note(
                    start=max(0., n.start - start),
                    end=min(n.end, end) - start,
                    pitch=n.pitch,
                    velocity=n.velocity)
                for n in all_notes
                if n.end >= start and n.start < end
            ]

            if len(notes) < min_notes_per_segment:
                continue

            if include_segment_id:
                yield ((file_id, i, i + bps), notes)
            else:
                yield notes


def normalize_tempo(midi, new_tempo=60):
    tempo_change_times, tempi = midi.get_tempo_changes()
    original_times = list(tempo_change_times) + [midi.get_end_time()]
    new_times = [original_times[0]]

    # Iterate through all the segments between the tempo changes.
    # Compute a new duration for each of them.
    for start, end, tempo in zip(original_times[:-1], original_times[1:], tempi):
        time = (end - start) * tempo / new_tempo
        new_times.append(new_times[-1] + time)

    midi.adjust_times(original_times, new_times)


def main(args):
    output = list(chop_midi(midis=args.fnames,
                            instrument_re=args.instrument_re,
                            bars_per_segment=args.bars_per_segment,
                            min_notes_per_segment=args.min_notes_per_segment,
                            include_segment_id=args.include_segment_id,
                            force_tempo=args.force_tempo,
                            skip_bars=args.skip_bars))
    with open(args.ofname, 'wb') as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    setup_parser(parser)
    main(parser.parse_args())
