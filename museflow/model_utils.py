import numpy as np
import tensorflow as tf


def create_train_op(optimizer, loss, variables, max_gradient_norm=None, name='training'):
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope(name):
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        return optimizer.apply_gradients(
            clip_gradients(grads_and_vars, max_gradient_norm),
            global_step=global_step)


def clip_gradients(grads_and_vars, max_gradient_norm):
    if max_gradient_norm is None:
        return grads_and_vars

    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return zip(clipped_gradients, variables)


class DatasetManager:

    def __init__(self, output_types=None, output_shapes=None):
        self._output_types = output_types
        self._output_shapes = output_shapes

        self.datasets = {}
        self._iterators = {}
        self._handles = {}
        self._handle_placeholder = tf.placeholder(tf.string, [], name='dataset_handle')
        self._global_iterator = None

    def add_dataset(self, name, dataset, one_shot=False):
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
        del self.datasets[name]
        del self._iterators[name]
        del self._handles[name]

    def initialize_dataset(self, session, name):
        session.run(self._iterators[name].initializer)

    def run(self, session, ops, dataset_name=None, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        if dataset_name is not None:
            if dataset_name not in self._handles:
                iterator = self._iterators[dataset_name]
                self._handles[dataset_name] = session.run(iterator.string_handle())
            feed_dict[self._handle_placeholder] = self._handles[dataset_name]
        return session.run(ops, feed_dict)

    def run_over_dataset(self, session, ops, dataset, feed_dict=None, batch_axis=None):
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
                results.append(self.run(session, ops, dataset_name, feed_dict=feed_dict))
            except tf.errors.OutOfRangeError:
                break

        if batch_axis is not None:
            # Flatten the structure of each item, concatenate the corresponding elements along
            # the batch axis and restore the structure.
            structure = results[0]
            results_flat = [tf.contrib.framework.nest.flatten(r) for r in results]
            results_flat = [np.concatenate(r, axis=batch_axis) for r in zip(*results_flat)]
            results = tf.contrib.framework.nest.pack_sequence_as(structure, results_flat)

        if dataset_name == '__tmp':
            self.remove_dataset(dataset_name)

        return results

    def get_batch(self):
        if self._global_iterator is None:
            self._global_iterator = tf.data.Iterator.from_string_handle(
                self._handle_placeholder, self._output_types, self._output_shapes)
        return self._global_iterator.get_next()


def prepare_train_and_val_data(train_generator, val_generator, output_types, output_shapes,
                               train_batch_size, val_batch_size, shuffle_buffer_size=100000,
                               num_epochs=None, dataset_manager=None):
    """A utility function to prepare a DatasetManager with training and validation data."""
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
    """A utility function to create a simple validation or test dataset."""
    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_generator(generator, output_types)
        if batch_size is not None:
            dataset = dataset.padded_batch(batch_size, output_shapes)
        return dataset
