import math
import re
import sys

import pretty_midi


def chop_midi(midis, instrument_re, bars_per_segment, min_notes_per_segment=2,
              include_segment_id=False):
    for file_index, midi in enumerate(midis):
        file_id = file_index
        if not isinstance(midi, pretty_midi.PrettyMIDI):
            file_id = str(midi)
            midi = pretty_midi.PrettyMIDI(midi)
        instruments = [i for i in midi.instruments if re.search(instrument_re, i.name)]
        if len(instruments) != 1:
            # TODO: Add support for multiple instruments?
            print(
                'Regex {!r} matches {} tracks in file {!r}; skipping file'
                .format(instrument_re, len(instruments), file_id), file=sys.stderr)
            continue
        [instrument] = instruments

        downbeats = midi.get_downbeats()
        b = bars_per_segment
        for i in range(0, len(downbeats), b):
            start = downbeats[i]
            end = downbeats[i + b] if i + b < len(downbeats) else midi.get_end_time()
            if math.isclose(start, end):
                continue

            # Find notes that overlap with this segment and clip them
            notes = [
                pretty_midi.Note(
                    start=max(0., n.start - start),
                    end=min(n.end, end) - start,
                    pitch=n.pitch,
                    velocity=n.velocity)
                for n in instrument.notes
                if n.end >= start and n.start < end
            ]

            if len(notes) < min_notes_per_segment:
                continue

            if include_segment_id:
                yield ((file_id, i, i+b), notes)
            else:
                yield notes
