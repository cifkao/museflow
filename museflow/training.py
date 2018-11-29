import os

import numpy as np
import tensorflow as tf


def create_train_op(optimizer, loss, variables, max_gradient_norm=None, name='training'):
    with tf.variable_scope(name):
        grads_and_vars = optimizer.compute_gradients(loss, variables)
        return optimizer.apply_gradients(
            clip_gradients(grads_and_vars, max_gradient_norm),
            global_step=tf.train.get_or_create_global_step())


def clip_gradients(grads_and_vars, max_gradient_norm):
    if max_gradient_norm is None:
        return grads_and_vars

    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return zip(clipped_gradients, variables)


class DatasetManager:

    def __init__(self, session):
        self._session = session

        self.datasets = {}
        self._iterators = {}
        self._handles = {}
        self._handle_placeholder = tf.placeholder(tf.string, [])
        self._global_iterator = None
        self.data_batch = None

    def add_dataset(self, name, dataset, one_shot=False):
        self.datasets[name] = dataset
        if one_shot:
            iterator = dataset.make_one_shot_iterator()
        else:
            iterator = dataset.make_initializable_iterator()
        self._iterators[name] = iterator
        self._handles[name] = self._session.run(iterator.string_handle())

        if self._global_iterator is None:
            self._global_iterator = tf.data.Iterator.from_string_handle(
                self._handle_placeholder, dataset.output_types, dataset.output_shapes)
            self.data_batch = self._global_iterator.get_next()

    def initialize_dataset(self, name):
        self._session.run(self._iterators[name].initializer)

    def run(self, ops, dataset_name=None, feed_dict=None):
        if feed_dict is None:
            feed_dict = {}
        if dataset_name is not None:
            feed_dict[self._handle_placeholder] = self._handles[dataset_name]
        return self._session.run(ops, feed_dict)

    def run_over_dataset(self, ops, dataset_name, feed_dict=None, batch_axis=None):
        self.initialize_dataset(dataset_name)
        results = []
        while True:
            try:
                results.append(self.run(ops, dataset_name))
            except tf.errors.OutOfRangeError:
                break

        if batch_axis is not None:
            # Flatten the structure of each item, concatenate the corresponding elements along
            # the batch axis and restore the structure.
            structure = results[0]
            results_flat = [tf.contrib.framework.nest.flatten(r) for r in results]
            results_flat = [np.concatenate(r, axis=batch_axis) for r in zip(*results_flat)]
            results = tf.contrib.framework.nest.pack_sequence_as(structure, results_flat)

        return results


class BasicTrainer:

    def __init__(self, session, logdir, logging_period, validation_period):
        self.session = session
        self._logdir = logdir
        self._logging_period = logging_period
        self._validation_period = validation_period

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._step = 0
        self._dataset_manager = DatasetManager(self.session)
        self._val_dataset_names = []
        self._has_data = False
        self._writer = None
        self._latest_saver = None
        self._best_saver = None

    @property
    def step(self):
        return self._step

    def prepare_data(self, train_dataset, validation_dataset):
        if self._has_data:
            raise RuntimeError('Data already prepared')

        self._dataset_manager.add_dataset('train', train_dataset, one_shot=True)
        self._dataset_manager.add_dataset('val', validation_dataset)
        self._has_data = True

        return self._dataset_manager.data_batch

    def train(self, train_op, loss, init_op=(), train_summary_op=()):
        if self._writer is None:
            self._writer = tf.summary.FileWriter(logdir=self._logdir, graph=tf.get_default_graph())
            with tf.name_scope('savers'):
                self._latest_saver = tf.train.Saver(name='latest')
                self._best_saver = tf.train.Saver(name='best', max_to_keep=1)

        self.session.run(init_op)

        best_mean_loss = np.inf
        def validate_and_save(loss):
            nonlocal best_mean_loss

            mean_loss = self.validate(loss)
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                self.save_best()
            self.save_latest()

        while True:
            print(self._step)
            if self._step % self._validation_period == 0:
                validate_and_save(loss)

            try:
                _, train_summary, train_loss = self._dataset_manager.run(
                    (train_op, train_summary_op, loss), 'train')
            except tf.errors.OutOfRangeError:
                break

            self._step = self.session.run(self._global_step_tensor)

            if self._step % self._logging_period == 0:
                if train_summary:
                    self._writer.add_summary(train_summary, self._step)

            if np.isnan(train_loss):
                print('NaN loss, stopping training')
                break

        validate_and_save(loss)

    def validate(self, loss, write_summaries=True):
        val_losses = self._dataset_manager.run_over_dataset(loss, 'val')
        mean_loss = np.mean(val_losses)

        if write_summaries:
            val_summary = tf.Summary(value=[
                tf.Summary.Value(tag='val/loss', simple_value=mean_loss)
            ])
            self._writer.add_summary(val_summary, self._step)

        return mean_loss

    def save_latest(self):
        self._latest_saver.save(
            self.session,
            os.path.join(self._logdir, 'latest.ckpt'),
            latest_filename='latest_checkpoint',
            global_step=self._step)

    def save_best(self):
        self._best_saver.save(
            self.session,
            os.path.join(self._logdir, 'best.ckpt'),
            latest_filename='best_checkpoint',
            global_step=self._step)
