# pylint: disable=attribute-defined-outside-init

import numpy as np
from numpy.testing import assert_equal
import tensorflow as tf

from museflow import model_utils


class TestDatasetManager:

    def setup(self):
        self.data = np.arange(18).reshape(6, 3)

        self.dm = model_utils.DatasetManager()
        self.dm.add_dataset('data', tf.data.Dataset.from_tensor_slices(self.data).batch(2))
        self.batch = self.dm.get_batch()
        self.sess = tf.Session()

    def test_run_over_dataset(self):
        fetches = (tf.reduce_max(self.batch), {'sum': tf.reduce_sum(self.batch)})
        result = self.dm.run_over_dataset(self.sess, fetches, 'data')
        assert_equal(result, ([5, 11, 17], {'sum': [15, 51, 87]}))

    def test_run_over_dataset_concat(self):
        fetches = [self.batch, (self.batch * 2, {'x': self.batch * 3})]
        expected_result = [list(self.data), (list(self.data * 2), {'x': list(self.data * 3)})]
        result = self.dm.run_over_dataset(self.sess, fetches, 'data', concat_batches=True)
        assert_equal(result, expected_result)

    def teardown(self):
        self.sess.close()
        tf.reset_default_graph()
