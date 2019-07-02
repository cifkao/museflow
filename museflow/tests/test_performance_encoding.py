# pylint: disable=attribute-defined-outside-init

import random
import warnings

import pytest
from numpy.testing import assert_equal
from pretty_midi import Note

from museflow.encodings import PerformanceEncoding


class TestPerformanceEncoding:

    def setup(self):
        random.seed(42)
        self.note_data = [
            [(0.0, 0.4, 29, 98), (0.7, 1.0, 29, 93), (1.3, 1.7, 29, 100), (2.0, 2.5, 29, 85),
             (2.7, 3.0, 33, 87), (3.3, 3.7, 36, 110), (4.0, 4.6, 30, 100), (4.7, 5.0, 30, 89),
             (5.7, 5.9, 30, 107), (6.0, 6.4, 38, 93), (6.7, 6.9, 38, 98), (7.0, 7.4, 30, 102),
             (8.0, 8.4, 29, 87), (8.7, 9.0, 33, 98), (9.0, 9.4, 36, 102)],
            [(0.0, 2.2, 70, 58), (0.0, 2.2, 62, 58), (0.0, 2.2, 65, 58), (0.0, 2.2, 68, 58),
             (0.0, 3.4, 46, 59), (2.5, 2.7, 70, 63), (2.5, 2.7, 62, 63), (2.5, 2.7, 65, 63),
             (2.5, 2.7, 68, 63), (3.5, 5.8, 67, 68), (3.5, 5.8, 58, 68), (3.5, 5.8, 61, 68),
             (3.5, 5.8, 63, 68), (3.6, 5.4, 51, 36), (5.7, 5.9, 75, 61), (5.8, 6.1, 73, 45),
             (5.9, 6.2, 70, 61), (6.0, 7.6, 67, 70), (6.5, 7.6, 70, 61)],
            [(0.0, 1.0, 60, 60), (0.4, 0.7, 61, 60), (0.4, 1.0, 62, 60), (0.5, 1.0, 63, 60),
             (0.8, 1.0, 64, 60)]
        ]

    @pytest.mark.parametrize(
        'encoding_args',
        [{'use_all_off_event': False},
         {'use_all_off_event': True},
         {'max_shift_units': 10}])
    def test_encode_decode(self, encoding_args):
        enc = PerformanceEncoding(use_velocity=True, velocity_unit=1, time_unit=0.01,
                                  warn_on_errors=True, **encoding_args)
        for notes in self.note_data:
            with warnings.catch_warnings(record=True) as w:
                result = enc.decode(enc.encode(_tuples_to_notes(notes)))
            assert not w
            assert_equal(sorted(_quantize(_notes_to_tuples(result))), sorted(_quantize(notes)))

    @pytest.mark.parametrize('as_ids', [True, False])
    def test_encode_shuffled(self, as_ids):
        enc = PerformanceEncoding(use_velocity=True, velocity_unit=1, time_unit=0.01)

        for notes in self.note_data:
            result = enc.encode(_tuples_to_notes(notes), as_ids=as_ids)
            for _ in range(2):
                shuffled_notes = random.sample(notes, len(notes))
                result2 = enc.encode(_tuples_to_notes(shuffled_notes), as_ids=as_ids)
                assert_equal(result2, result)

    @pytest.mark.parametrize(
        'expected_num_warnings, events',
        [(1, [('NoteOn', 10), ('TimeShift', 5)]),
         (2, [('NoteOn', 10), ('TimeShift', 5), ('NoteOff', 11)]),
         (1, [('NoteOn', 10), ('TimeShift', 5), ('NoteOff', 10), ('NoteOff', '*')]),
         (1, [('NoteOn', 10), ('TimeShift', 5), ('NoteOff', '*'), ('NoteOff', 10)]),
         (0, [('NoteOn', 10), ('TimeShift', 5), ('NoteOff', '*')])])
    def test_decode_errors(self, events, expected_num_warnings):
        enc = PerformanceEncoding(use_velocity=False, time_unit=0.01, max_shift_units=20,
                                  use_all_off_event=True, warn_on_errors=True)
        with warnings.catch_warnings(record=True) as w:
            enc.decode(events)
        assert len(w) == expected_num_warnings


def _notes_to_tuples(notes):
    return [(n.start, n.end, n.pitch, n.velocity) for n in notes]


def _tuples_to_notes(tuples):
    return [Note(start=s, end=e, pitch=p, velocity=v) for s, e, p, v in tuples]


def _quantize(note_tuples):
    quantized = []
    for note in note_tuples:
        start, end, pitch, velocity = note
        quantized.append((int(start * 100 + .5), int(end * 100 + .5), pitch, velocity))
    return quantized
