import os
import types

import numpy as np
import tensorflow as tf

from museflow import logger
from museflow.config import configurable


@configurable(['latest_saver', 'best_saver'])
class BasicTrainer:
    """A class implementing a basic training/validation loop, model saving and model loading."""

    def __init__(self, dataset_manager, logdir, logging_period, validation_period=None,
                 session=None, training_placeholder=None, train_dataset_name='train',
                 val_dataset_name='val'):
        self.session = session or tf.Session()
        self._dataset_manager = dataset_manager
        self._logdir = logdir
        self._logging_period = logging_period
        self._validation_period = validation_period
        self._training_placeholder = training_placeholder
        self._train_dataset_name = train_dataset_name
        self._val_dataset_name = val_dataset_name

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._step = 0

        if self._training_placeholder is None:
            self._training_placeholder = tf.placeholder_with_default(False, [], name='is_training')

        self._writer = tf.summary.FileWriter(logdir=self._logdir, graph=tf.get_default_graph())
        with tf.name_scope('savers'):
            self._latest_saver = self._cfg.configure('latest_saver', tf.train.Saver,
                                                     name='latest', max_to_keep=2)
            self._best_saver = self._cfg.configure('best_saver', tf.train.Saver,
                                                   name='best', max_to_keep=1)

    @property
    def step(self):
        return self._step

    def train(self, train_op, loss, init_op=(), train_summary_op=()):
        train_generator = self.iter_train(train_op, loss, init_op, train_summary_op, period=None)
        return next(train_generator)

    def iter_train(self, train_op, loss, init_op=(), train_summary_op=(), period=0):
        """Return a generator that runs the training loop.

        Every `period` training steps, the generator yields an object with the attributes `step`,
        `train_loss`, `last_val_loss`, `best_val_loss` and `best_val_step`. Note that the first
        time, `step` will be 0 and the loss values possibly unknown.

        If `period` is not specified, `validation_period` is used instead.
        """
        if period == 0:
            period = self._validation_period

        self.session.run(init_op)

        state = types.SimpleNamespace(step=self._step,
                                      train_loss=None,
                                      last_val_loss=None,
                                      best_val_loss=np.inf,
                                      best_val_step=-1)

        def validate_and_save():
            state.last_val_loss = self.validate(loss, write_summaries=True)
            if state.last_val_loss < state.best_val_loss:
                state.best_val_loss, state.best_val_step = state.last_val_loss, self._step
                self.save_variables('best')
            self.save_variables('latest')

        while True:
            if self._validation_period is not None and self._step % self._validation_period == 0:
                validate_and_save()
            if period is not None and self._step % period == 0:
                yield types.SimpleNamespace(**state.__dict__)

            try:
                state.train_loss, _ = self.training_step(train_op, loss, train_summary_op)
                state.step = self._step
            except tf.errors.OutOfRangeError:
                logger.info(f'Training finished after {state.step} steps, loss: {state.train_loss}')
                break

            if np.isnan(state.train_loss):
                logger.error('NaN loss, stopping training')
                break

        if self._validation_period is not None:
            validate_and_save()
        yield state

    def training_step(self, train_op, loss, train_summary_op=(), feed_dict=None,
                      write_summaries=True, log=True):
        if feed_dict is None:
            feed_dict = {}

        _, train_summary, train_loss = self._dataset_manager.run(
            self.session, (train_op, train_summary_op, loss), self._train_dataset_name,
            feed_dict={self._training_placeholder: True, **feed_dict})

        self._step = self.session.run(self._global_step_tensor)

        if self._step % self._logging_period == 0:
            if write_summaries and train_summary:
                self._writer.add_summary(train_summary, self._step)
            if log:
                logger.info('step: {}, loss: {}'.format(self._step, train_loss))

        return train_loss, train_summary

    def validate(self, loss, write_summaries=False):
        val_losses = self._dataset_manager.run_over_dataset(self.session, loss,
                                                            self._val_dataset_name)
        mean_loss = np.mean(val_losses)

        if write_summaries:
            self.write_scalar_summary('{}/loss'.format(self._val_dataset_name), mean_loss)

        return mean_loss

    def save_variables(self, checkpoint_name='latest'):
        saver = self._best_saver if checkpoint_name == 'best' else self._latest_saver
        saver.save(
            self.session,
            os.path.join(self._logdir, checkpoint_name + '.ckpt'),
            latest_filename=checkpoint_name + '_checkpoint',
            global_step=self._step)

    def load_variables(self, checkpoint_name='latest', checkpoint_file=None):
        saver = self._best_saver if checkpoint_name == 'best' else self._latest_saver
        if not checkpoint_file:
            checkpoint_file = tf.train.latest_checkpoint(self._logdir,
                                                         checkpoint_name + '_checkpoint')
        saver.restore(self.session, checkpoint_file)
        logger.info('Variables restored from {}'.format(checkpoint_file))
        self._step = self.session.run(self._global_step_tensor)

    def write_scalar_summary(self, name, value, step=None):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=value)
        ])
        self._writer.add_summary(summary, step if step is not None else self._step)
