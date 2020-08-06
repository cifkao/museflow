from note_seq.protobuf import music_pb2
import numpy as np
import pretty_midi


class PianoRollEncoding:

    def __init__(self, sampling_frequency, normalize=True, binarize=False, min_pitch=0,
                 max_pitch=127, dtype=np.float32):
        self._fs = sampling_frequency
        self._normalize = normalize
        self._binarize = binarize
        self._min_pitch = min_pitch
        self._max_pitch = max_pitch
        self._dtype = dtype

        self.num_rows = max_pitch - min_pitch + 1

    def encode(self, notes):
        if isinstance(notes, music_pb2.NoteSequence):
            notes = [pretty_midi.Note(start=n.start_time, end=n.end_time,
                                      pitch=n.pitch, velocity=n.velocity)
                     for n in notes.notes if not n.is_drum]

        instrument = pretty_midi.Instrument(0)
        instrument.notes[:] = notes
        roll = instrument.get_piano_roll(fs=self._fs)
        roll = roll[self._min_pitch:self._max_pitch + 1]

        if self._binarize:
            roll = roll > 1e-9
        elif self._normalize:
            roll /= 127.

        return roll.astype(self._dtype)
