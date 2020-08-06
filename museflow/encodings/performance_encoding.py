from collections import defaultdict
import heapq
import warnings

from note_seq.protobuf import music_pb2
from note_seq.constants import STANDARD_PPQ
import numpy as np
import pretty_midi

from museflow import logger
from museflow.vocabulary import Vocabulary


class PerformanceEncoding:
    """An encoding of note sequences based on Magenta's PerformanceRNN.

    This is actually very similar to how MIDI works.
    See https://magenta.tensorflow.org/performance-rnn.
    """

    def __init__(self, time_unit=0.01, max_shift_units=100, velocity_unit=4, use_velocity=True,
                 use_all_off_event=False, use_drum_events=False, use_magenta=False, errors='remove',
                 warn_on_errors=False):

        self._time_unit = time_unit
        self._max_shift_units = max_shift_units
        self._velocity_unit = velocity_unit
        self._use_velocity = use_velocity
        self._use_all_off_event = use_all_off_event
        self._use_drum_events = use_drum_events
        self._use_magenta = use_magenta
        self._errors = errors
        self._warn_on_errors = warn_on_errors

        if use_drum_events:
            assert use_magenta

        max_velocity_units = (128 + velocity_unit - 1) // velocity_unit

        wordlist = (['<pad>', '<s>', '</s>'] +
                    [('NoteOn', i) for i in range(128)] +
                    [('NoteOff', i) for i in range(128)] +
                    ([('NoteOff', '*')] if use_all_off_event else []) +
                    ([('DrumOn', i) for i in range(128)] +
                     [('DrumOff', i) for i in range(128)]
                     if use_drum_events else []) +
                    [('TimeShift', i + 1) for i in range(max_shift_units)])

        if use_velocity:
            wordlist.extend([('SetVelocity', i + 1) for i in range(max_velocity_units)])
            self._default_velocity = 0
        else:
            self._default_velocity = 127

        self.vocabulary = Vocabulary(wordlist)

    def encode(self, notes, as_ids=True, add_start=False, add_end=False):
        is_drum = False
        if isinstance(notes, music_pb2.NoteSequence):
            is_drum = (len(notes.notes) > 0 and notes.notes[0].is_drum)
            notes = [pretty_midi.Note(start=n.start_time, end=n.end_time,
                                      pitch=n.pitch, velocity=n.velocity)
                     for n in notes.notes]

        queue = _NoteEventQueue(notes, quantization_step=self._time_unit)
        events = [self.vocabulary.start_token] if add_start else []

        last_t = 0
        velocity = self._default_velocity
        for t, note, is_onset in queue:
            while last_t < t:
                shift_amount = min(t - last_t, self._max_shift_units)
                last_t += shift_amount
                events.append(('TimeShift', shift_amount))

            if is_onset:
                note_velocity = note.velocity
                if note_velocity > 127 or note_velocity < 1:
                    warnings.warn(f'Invalid velocity value: {note_velocity}')
                    note_velocity = self._default_velocity
                if velocity != note_velocity:
                    velocity = note_velocity
                    if self._use_velocity:
                        events.append(('SetVelocity', velocity // self._velocity_unit + 1))
                if is_drum and self._use_drum_events:
                    events.append(('DrumOn', note.pitch))
                else:
                    events.append(('NoteOn', note.pitch))
            else:
                if is_drum and self._use_drum_events:
                    events.append(('DrumOff', note.pitch))
                else:
                    events.append(('NoteOff', note.pitch))

        if self._use_all_off_event:
            events = _compress_note_offs(events)

        if add_end:
            events.append(self.vocabulary.end_token)

        if as_ids:
            return self.vocabulary.to_ids(events)
        return events

    def decode(self, tokens):
        notes = []
        notes_on = defaultdict(list)
        error_count = 0
        is_drum = False

        t = 0
        velocity = self._default_velocity
        for token in tokens:
            if isinstance(token, (int, np.integer)):
                token = self.vocabulary.from_id(token)
            if token not in self.vocabulary:
                raise RuntimeError(f'Invalid token: {token}')
            if not isinstance(token, tuple):
                continue
            event, value = token

            if event == 'TimeShift':
                t += value * self._time_unit
            elif event == 'SetVelocity':
                velocity = (value - 1) * self._velocity_unit
            elif event in ['NoteOn', 'DrumOn']:
                note = pretty_midi.Note(start=t, end=None, pitch=value, velocity=velocity)
                notes.append(note)
                notes_on[value].append(note)
                is_drum |= (event == 'DrumOn')
            elif event in ['NoteOff', 'DrumOff']:
                if value == '*':
                    assert self._use_all_off_event

                    if not any(notes_on.values()):
                        error_count += 1

                    for note_list in notes_on.values():
                        for note in note_list:
                            note.end = t
                        note_list.clear()
                else:
                    try:
                        note = notes_on[value].pop()
                        note.end = t
                    except IndexError:
                        error_count += 1

        if error_count:
            self._log_errors('Encountered {} errors'.format(error_count))

        if any(notes_on.values()):
            if self._errors == 'remove':
                self._log_errors('Removing {} hanging note(s)'.format(
                    sum(len(l) for l in notes_on.values())))
                for notes_on_list in notes_on.values():
                    for note in notes_on_list:
                        notes.remove(note)
            else:  # 'ignore'
                self._log_errors('Ignoring {} hanging note(s)'.format(
                    sum(len(l) for l in notes_on.values())))

        if self._use_magenta:
            sequence = music_pb2.NoteSequence()
            sequence.ticks_per_quarter = STANDARD_PPQ
            for note0 in notes:
                note = sequence.notes.add()
                note.start_time = note0.start
                note.end_time = note0.end
                note.pitch = note0.pitch
                note.velocity = note0.velocity
                note.is_drum = is_drum
            return sequence

        return notes

    def _log_errors(self, message):
        if self._warn_on_errors:
            warnings.warn(message, RuntimeWarning)
        else:
            logger.debug(message)


class _NoteEventQueue:
    """
    A priority queue of note onsets and offsets.

    The queue is ordered according to time and pitch.
    Offsets come before onsets that occur at the same time, unless they correspond
    to the same note.
    """

    def __init__(self, notes, quantization_step=None):
        """Initialize the queue.

        Args:
            notes: A list of `pretty_midi.Note` objects to fill the queue with.
            quantization_step: The quantization step in seconds. If `None`, no
                quantization will be performed.
        """
        self._quantization_step = quantization_step

        # Build a heap of note onsets and offsets. For now, we only add the onsets;
        # an offset is added once the corresponding onset is popped. This is an easy
        # way to make sure that we never pop the offset first.
        # Below, the ID of the Note object is used to stop the heap algorithm from
        # comparing the Note itself, which would raise an exception.
        self._heap = [(self._quantize(note.start), True, note.pitch, id(note), note)
                      for note in notes]
        heapq.heapify(self._heap)

    def pop(self):
        """Return the next event from the queue.

        Returns:
            A tuple of the form `(time, note, is_onset)` where `time` is the time of the
            event (expressed as the number of quantization steps if applicable) and `note`
            is the corresponding `Note` object.
        """
        time, is_onset, _, _, note = heapq.heappop(self._heap)
        if is_onset:
            # Add the offset to the queue
            heapq.heappush(self._heap,
                           (self._quantize(note.end), False, note.pitch, hash(note), note))

        return time, note, is_onset

    def __iter__(self):
        while self._heap:
            yield self.pop()

    def _quantize(self, value):
        if self._quantization_step:
            return int(value / self._quantization_step + 0.5)  # Round to nearest int
        return value


def _compress_note_offs(tokens):
    """Given a sequence of events, convert single-note-off events to all-note-off events.

    It is assumed that the sequence is correct, i.e. generated by `PerformanceEncoding.encode`.
    """
    new_tokens = []
    note_offs = []  # note-offs waiting to be added
    num_notes_on = 0
    for token in tokens:
        event, value = token if isinstance(token, tuple) else (token, None)

        if note_offs and event != 'NoteOff':
            if num_notes_on == 0:
                new_tokens.append(('NoteOff', '*'))
            else:
                new_tokens.extend(note_offs)
            note_offs.clear()

        if event == 'NoteOn':
            num_notes_on += 1
        if event == 'NoteOff':
            assert isinstance(value, int), 'NoteOff pitch must be an int'
            note_offs.append(token)
            num_notes_on -= 1
            assert num_notes_on >= 0, 'Invalid sequence: found NoteOff with no matching NoteOn'
            continue

        new_tokens.append(token)

    assert num_notes_on == 0, 'Invalid sequence: found NoteOn with no matching NoteOff'
    if note_offs:
        new_tokens.append(('NoteOff', '*'))

    return new_tokens
