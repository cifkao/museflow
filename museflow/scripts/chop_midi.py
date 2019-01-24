"""Chop midi files into segments containing a given number of bars."""

import argparse
import collections
import math
import pickle
import re

import pretty_midi

from museflow import logger


def setup_argparser(parser):
    parser.set_defaults(func=main)
    parser.add_argument('input_files', type=argparse.FileType('rb'), nargs='+', metavar='FILE')
    parser.add_argument('output_file', type=argparse.FileType('wb'), metavar='OUTPUTFILE')
    parser.add_argument('-i', '--instrument-re', type=str, default='.*')
    parser.add_argument('-b', '--bars-per-segment', type=lambda l: [int(x) for x in l.split(',')],
                        default=[8])
    parser.add_argument('-n', '--min-notes-per-segment', type=int, default=1)
    parser.add_argument('-t', '--force-tempo', type=float, default=None)
    parser.add_argument('--skip-bars', type=int, default=0)
    parser.add_argument('--include-segment-id', action='store_true')


def chop_midi(files, instrument_re, bars_per_segment, min_notes_per_segment=1,
              include_segment_id=False, force_tempo=None, skip_bars=0):
    if isinstance(bars_per_segment, int):
        bars_per_segment = [bars_per_segment]
    bars_per_segment = list(bars_per_segment)

    for file in files:
        file_id = file.name
        midi = pretty_midi.PrettyMIDI(file)

        if force_tempo is not None:
            normalize_tempo(midi, force_tempo)

        instruments = [i for i in midi.instruments if re.search(instrument_re, i.name)]
        if not instruments:
            logger.warning('Regex {} does not match any track in file {}; skipping file'.format(
                instrument_re, file_id))
            continue
        all_notes = [n for i in instruments for n in i.notes]
        all_notes.sort(key=lambda n: n.start)

        def is_overlapping(note, start, end):
            """Check whether the given note overlaps with the given interval."""
            return ((note.end > start or math.isclose(note.end, start)) and
                    (note.start < end and not math.isclose(note.start, end)))

        downbeats = midi.get_downbeats()[skip_bars:]
        for bps in bars_per_segment:
            note_queue = collections.deque(all_notes)
            notes = []  # notes in the current segment
            for i in range(0, len(downbeats), bps):
                start = downbeats[i]
                end = downbeats[i + bps] if i + bps < len(downbeats) else midi.get_end_time()
                if math.isclose(start, end):
                    continue

                # Filter the notes from the previous segment to keep those that overlap with the
                # current one.
                notes[:] = (n for n in notes if is_overlapping(n, start, end))

                # Add new overlapping notes. note_queue is sorted by onset time, so we can stop
                # after the first note which is outside the segment.
                while note_queue and is_overlapping(note_queue[0], start, end):
                    notes.append(note_queue.popleft())

                # Clip the notes to the segment.
                notes_clipped = [
                    pretty_midi.Note(
                        start=max(0., n.start - start),
                        end=min(n.end, end) - start,
                        pitch=n.pitch,
                        velocity=n.velocity)
                    for n in notes
                ]

                if len(notes_clipped) < min_notes_per_segment:
                    continue

                if include_segment_id:
                    yield ((file_id, i, i + bps), notes_clipped)
                else:
                    yield notes_clipped


def normalize_tempo(midi, new_tempo=60):
    if math.isclose(midi.get_end_time(), 0.):
        return

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
    output = list(chop_midi(files=args.input_files,
                            instrument_re=args.instrument_re,
                            bars_per_segment=args.bars_per_segment,
                            min_notes_per_segment=args.min_notes_per_segment,
                            include_segment_id=args.include_segment_id,
                            force_tempo=args.force_tempo,
                            skip_bars=args.skip_bars))
    pickle.dump(output, args.output_file)
