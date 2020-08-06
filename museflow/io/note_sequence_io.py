"""Utilities for manipulating databases of Magenta NoteSequence objects."""
import random

import lmdb
from note_seq.protobuf import music_pb2


_NO_DEFAULT = object()


class NoteSequenceDB:
    """A LMDB database of Magenta note sequences."""

    def __init__(self, path, write=False, **kwargs):
        self._db_kwargs = dict(path=path, subdir=False, lock=False, readonly=not write)
        self._db_kwargs.update(kwargs)
        self.db = None

    def __enter__(self):
        if self.db is not None:
            raise RuntimeError('Database already open')
        self.db = lmdb.Environment(**self._db_kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        db = self.db
        self.db = None
        return db.__exit__(exc_type, exc_value, traceback)

    def begin(self, write=None):
        """Begin a transaction.

        Returns a NoteSequenceDBTransaction object.
        """
        if self.db is None:
            raise RuntimeError('Database not open')
        if write is None:
            write = not self._db_kwargs['readonly']
        return NoteSequenceDBTransaction(self.db.begin(buffers=True, write=write))


class NoteSequenceDBTransaction:

    def __init__(self, txn):
        self.txn = txn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.txn.__exit__(exc_type, exc_value, traceback)

    def get(self, key, default=_NO_DEFAULT):
        val = self.txn.get(key.encode())
        if val is None:
            if default is _NO_DEFAULT:
                raise KeyError(key)
            else:
                return default
        return music_pb2.NoteSequence.FromString(val)

    def put(self, key, val):
        self.txn.put(key.encode(), val.SerializeToString())

    def pop(self, key):
        self.txn.pop(key.encode())

    def __iter__(self):
        for key in self.txn.cursor().iternext(keys=True, values=False):
            yield bytes(key).decode()

    def items(self):
        for key, val in self.txn.cursor():
            yield bytes(key).decode(), music_pb2.NoteSequence.FromString(val)


def save_sequences_db(sequences, keys, db_path):
    """Store the given note sequences in a LMDB database."""
    sequences = list(sequences)
    keys = list(keys)
    if len(keys) != len(sequences):
        raise RuntimeError('Keys and sequences are not the same length.')

    # Estimate database size
    sample = random.Random(42).sample(sequences, min(200, len(sequences)))
    avg_size = sum(len(seq.SerializeToString()) for seq in sample) / len(sample)
    map_size = 8 * avg_size * (len(sequences) + 1)

    with lmdb.open(db_path, subdir=False, lock=False, map_size=map_size) as db:
        with db.begin(buffers=True, write=True) as txn:
            for key, seq in zip(keys, sequences):
                if not txn.put(key.encode(), seq.SerializeToString(), overwrite=False):
                    raise RuntimeError(f'Duplicate key: {key}')
