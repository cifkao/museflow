import random

import numpy as np
import tensorflow as tf

from museflow import logger
from museflow.config import configurable


@configurable(['optimizer', 'lr_decay'])
def create_train_op(cfg, loss, optimizer=None, variables=None, max_gradient_norm=None,
                    name='training'):
    """Create a training op."""
    global_step = tf.train.get_or_create_global_step()

    if optimizer is None:
        opt_args = {}
        learning_rate = cfg['lr_decay'].maybe_configure(global_step=global_step)
        if learning_rate is not None:
            opt_args['learning_rate'] = learning_rate
            tf.summary.scalar('learning_rate', learning_rate, family='train')

        optimizer = cfg['optimizer'].configure(tf.train.AdamOptimizer, **opt_args)

    if variables is None:
        variables = tf.trainable_variables()

    logger.info("'{}' op trains {} variables:\n\n{}\n".format(
        name, len(variables), summarize_variables(variables)))

    with tf.variable_scope(name):
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        return optimizer.apply_gradients(
            clip_gradients(grads_and_vars, max_gradient_norm),
            global_step=global_step)


def clip_gradients(grads_and_vars, max_gradient_norm):
    """Perform gradient clipping by global norm."""
    if max_gradient_norm is None:
        return grads_and_vars

    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return zip(clipped_gradients, variables)


def summarize_variables(variables):
    """Return a string with a table of names and shapes of the given variables."""
    cols = [['Name'], ['Shape']]
    for var in variables:
        shape = var.get_shape().as_list()
        cols[0].append(var.name)
        cols[1].append(str(shape))
    widths = [max(len(x) for x in col) for col in cols]

    lines = ['  '.join(text.ljust(widths[i]) for i, text in enumerate(row)) for row in zip(*cols)]
    lines.insert(1, '  '.join('-' * w for w in widths))
    return '\n'.join(lines)


class DatasetManager:
    """A class for managing TensorFlow datasets.

    A `DatasetManager` objects holds a collection of datasets (typically, a training set and a
    validation set) and their iterators and takes care of switching between them when running
    TensorFlow ops.
    """

    def __init__(self, output_types=None, output_shapes=None):
        self._output_types = output_types
        self._output_shapes = output_shapes

        self.datasets = {}
        self._iterators = {}
        self._handles = {}
        self._handle_placeholder = tf.placeholder(tf.string, [], name='dataset_handle')
        self._global_iterator = None
        self._last_session = None

    def add_dataset(self, name, dataset, one_shot=False):
        """Add a new dataset to the collection.

        Args:
            name: A name for the dataset (e.g. `'train'`, `'val'`).
            dataset: An instance of `tf.data.Dataset`.
            one_shot: If `True`, a one-shot iterator will be used (appropriate for training
                datasets); if `False` (default), an initializable iterator will be used
                (requiring to call `initialize_dataset`).
        """
        self.datasets[name] = dataset
        if one_shot:
            self._iterators[name] = dataset.make_one_shot_iterator()
        else:
            self._iterators[name] = dataset.make_initializable_iterator()

        if self._output_types is None:
            self._output_types = dataset.output_types
        if self._output_shapes is None:
            self._output_shapes = dataset.output_shapes

    def remove_dataset(self, name):
        """Remove the given dataset from the collection."""
        del self.datasets[name]
        del self._iterators[name]
        del self._handles[name]

    def initialize_dataset(self, session, name):
        """Initialize the given dataset's iterator."""
        session.run(self._iterators[name].initializer)

    def run(self, session, fetches, dataset_name=None, feed_dict=None):
        """Run the given TensorFlow ops while using the chosen dataset.

        Args:
            session: A TensorFlow `Session`.
            fetches: The `fetches` to pass to `session.run`.
            dataset_name: The name of the dataset to use. If `None` (default), no dataset will be
                used.
            feed_dict: The `feed_dict` to pass to `session.run`.
        Returns:
            The return value of `session.run`.
        """
        if feed_dict is None:
            feed_dict = {}
        if dataset_name is not None:
            if session is not self._last_session:
                self._handles.clear()

            if dataset_name not in self._handles:
                iterator = self._iterators[dataset_name]
                self._handles[dataset_name] = session.run(iterator.string_handle())
            feed_dict[self._handle_placeholder] = self._handles[dataset_name]
        self._last_session = session
        return session.run(fetches, feed_dict)

    def run_over_dataset(self, session, fetches, dataset, feed_dict=None, concat_batches=False):
        """Run the given TensorFlow ops while iterating over an entire dataset.

        Args:
            session: A TensorFlow `Session`.
            fetches: The `fetches` to pass to `session.run`.
            dataset: The name of the dataset to use, or a new `tf.data.Dataset`.
            feed_dict: The `feed_dict` to pass to `session.run`.
            concat_batches: If `True`, the results will be concatenated along the first dimension.
        Returns:
            A list of results of `session.run`. If `fetches` is a nested structure of tensors,
            then the same nested structure will be returned, containing a list of results for each
            tensor. If `concat_batches` is `True`, the return value will be a list (or a nested
            structure of lists) obtained by concatenating all batches.
        """
        if isinstance(dataset, str):
            dataset_name = dataset
        else:
            # A dataset object was passed directly; add it temporarily, then remove it.
            dataset_name = '__tmp'
            self.add_dataset(dataset_name, dataset)

        self.initialize_dataset(session, dataset_name)
        results = []
        while True:
            try:
                results.append(self.run(session, fetches, dataset_name, feed_dict=feed_dict))
            except tf.errors.OutOfRangeError:
                break

        # Flatten the structure of each batch, put the corresponding elements together and restore
        # the structure.
        structure = results[0]
        results_flat = zip(*(tf.contrib.framework.nest.flatten(r) for r in results))
        if concat_batches:
            # We do not use np.concatenate since the shapes of the batches might be incompatible.
            # Instead, we stack the items of all batches in a list.
            results_flat = [[x for batch in r for x in batch] for r in results_flat]
        else:
            results_flat = [list(r) for r in results_flat]
        results = tf.contrib.framework.nest.pack_sequence_as(structure, results_flat)

        if dataset_name == '__tmp':
            self.remove_dataset(dataset_name)

        return results

    def get_batch(self):
        """Return a nested struture of tensors representing the next element of a dataset."""
        if self._global_iterator is None:
            self._global_iterator = tf.data.Iterator.from_string_handle(
                self._handle_placeholder, self._output_types, self._output_shapes)
        return self._global_iterator.get_next()


def prepare_train_and_val_data(train_generator, val_generator, output_types, output_shapes,
                               train_batch_size, val_batch_size, shuffle_buffer_size=100000,
                               num_epochs=None, dataset_manager=None):
    """Prepare a DatasetManager with training and validation data."""
    with tf.name_scope('train'):
        train_dataset = tf.data.Dataset.from_generator(train_generator, output_types)
        train_dataset = train_dataset.shuffle(
            shuffle_buffer_size, reshuffle_each_iteration=True).repeat(num_epochs)
        train_dataset = train_dataset.padded_batch(
            train_batch_size, output_shapes)
    if dataset_manager:
        dataset_manager.add_dataset('train', train_dataset, one_shot=True)

    val_dataset = make_simple_dataset(
        val_generator, output_types, output_shapes, val_batch_size, 'val')
    if dataset_manager:
        dataset_manager.add_dataset('val', val_dataset)

    return train_dataset, val_dataset


def make_simple_dataset(generator, output_types, output_shapes, batch_size=None, name='dataset'):
    """Create a simple validation or test dataset."""
    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_generator(generator, output_types)
        if batch_size is not None:
            dataset = dataset.padded_batch(batch_size, output_shapes)
        return dataset


def set_random_seed(seed):
    if seed is not None:
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
