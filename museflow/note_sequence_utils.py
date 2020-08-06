import copy as copy_lib
import math
import re
import sys
import warnings

from note_seq.protobuf import music_pb2
from note_seq import sequences_lib, midi_io
import numpy as np


def filter_sequence(sequence, instrument_re=None, instrument_ids=None, programs=None, drums=None,
                    copy=False):
    """Filter a Magenta `NoteSequence` in place.

    Args:
        sequence: The `NoteSequence` protobuffer to filter.
        instrument_re: A regular expression used to match instrument names.
        instrument_ids: A list of instrument IDs to match or `None` to match any ID.
        programs: A list of MIDI programs to match or `None` to match any program.
        drums: Include only drums (`True`) or only non-drums (`False`). If `None` (default), include
            both drums and non-drums.
        copy: If `True`, a copy of the sequence will be returned and the original sequence will
            be left unchanged.

    Returns:
        The filtered `NoteSequence`.
    """
    if copy:
        sequence, original_sequence = music_pb2.NoteSequence(), sequence
        sequence.CopyFrom(original_sequence)

    if isinstance(instrument_re, str):
        instrument_re = re.compile(instrument_re)

    # Filter the instruments based on name and ID
    deleted_ids = set()
    if instrument_re is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if not instrument_re.search(i.name))
    if instrument_ids is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if i.instrument not in instrument_ids)
    new_infos = [i for i in sequence.instrument_infos if i.instrument not in deleted_ids]
    del sequence.instrument_infos[:]
    for info in new_infos:
        sequence.instrument_infos.add().CopyFrom(info)

    # Filter the event collections
    for collection in [sequence.notes, sequence.pitch_bends, sequence.control_changes]:
        collection_copy = list(collection)
        del collection[:]

        for event in collection_copy:
            if event.instrument in deleted_ids:
                continue
            if instrument_ids is not None and event.instrument not in instrument_ids:
                continue
            if programs is not None and event.program not in programs:
                continue
            if drums is not None and event.is_drum != drums:
                continue
            collection.add().CopyFrom(event)

    return sequence


def normalize_tempo(sequence, new_tempo=60):
    if math.isclose(sequence.total_time, 0.):
        return copy_lib.deepcopy(sequence)

    tempo_change_times, tempi = zip(*sorted(
        (tempo.time, tempo.qpm) for tempo in sequence.tempos if tempo.time < sequence.total_time))
    original_times = list(tempo_change_times) + [sequence.total_time]
    new_times = [original_times[0]]

    # Iterate through all the intervals between the tempo changes.
    # Compute a new duration for each of them.
    for start, end, tempo in zip(original_times[:-1], original_times[1:], tempi):
        time = (end - start) * tempo / new_tempo
        new_times.append(new_times[-1] + time)

    def time_func(t):
        return np.interp(t, original_times, new_times)

    adjusted_sequence, skipped_notes = sequences_lib.adjust_notesequence_times(sequence, time_func)
    if skipped_notes:
        warnings.warn(f'{skipped_notes} notes skipped in adjust_notesequence_times', RuntimeWarning)

    del adjusted_sequence.tempos[:]
    tempo = adjusted_sequence.tempos.add()
    tempo.time = 0.
    tempo.qpm = new_tempo

    return adjusted_sequence


def split_on_downbeats(sequence, bars_per_segment, downbeats=None, skip_bars=0,
                       min_notes_per_segment=0, include_span=False):
    if downbeats is None:
        downbeats = midi_io.sequence_proto_to_pretty_midi(sequence).get_downbeats()
    downbeats = [d for d in downbeats if d < sequence.total_time]

    try:
        iter(bars_per_segment)
    except TypeError:
        bars_per_segment = [bars_per_segment]

    for bps in bars_per_segment:
        first_split = skip_bars or bps  # Do not split at time 0
        split_times = list(downbeats[first_split::bps])
        segments = sequences_lib.split_note_sequence(sequence,
                                                     hop_size_seconds=split_times)
        if skip_bars:
            # The first segment will contain the bars we want to skip
            segments.pop(0)

        for i, segment in enumerate(segments):
            start = skip_bars + i * bps
            end = start + bps

            if len(segment.notes) < min_notes_per_segment:
                print(f'Skipping segment {start}-{end} with {len(segment.notes)} notes',
                      file=sys.stderr)
                continue

            if include_span:
                yield start, end, segment
            else:
                yield segment


def get_downbeats(sequence):
    downbeats = midi_io.sequence_proto_to_pretty_midi(sequence).get_downbeats()
    return [d for d in downbeats if d < sequence.total_time]


def set_note_fields(sequence, **kwargs):
    for note in sequence.notes:
        for attr, val in kwargs.items():
            setattr(note, attr, val)
