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
                 training_ops=None, session=None, write_summaries=True,
                 train_dataset_name='train', val_dataset_name='val'):
        self.session = session or tf.Session()
        self._dataset_manager = dataset_manager
        self._logdir = logdir
        self._ops = training_ops or BasicTrainer.TrainingOps(loss=None, train_op=())
        self._logging_period = logging_period
        self._validation_period = validation_period
        self._train_dataset_name = train_dataset_name
        self._val_dataset_name = val_dataset_name

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._step = 0

        if self._ops.training_placeholder is None:
            self._ops.training_placeholder = tf.placeholder_with_default(False, [],
                                                                         name='is_training')

        if write_summaries:
            self._writer = tf.summary.FileWriter(logdir=self._logdir, graph=tf.get_default_graph())
        else:
            self._writer = None
        with tf.name_scope('savers'):
            self._latest_saver = self._cfg['latest_saver'].configure(tf.train.Saver,
                                                                     name='latest', max_to_keep=2)
            self._best_saver = self._cfg['best_saver'].configure(tf.train.Saver,
                                                                 name='best', max_to_keep=1)

    @property
    def step(self):
        return self._step

    def train(self):
        train_generator = self.iter_train(period=None)
        return next(train_generator)

    def iter_train(self, period=0):
        """Return a generator that runs the training loop.

        Every `period` training steps, the generator yields an object with the attributes `step`,
        `train_loss`, `last_val_loss`, `best_val_loss` and `best_val_step`. Note that the first
        time, `step` will be 0 and the loss values possibly unknown.

        If `period` is not specified, `validation_period` is used instead.
        """
        if period == 0:
            period = self._validation_period

        self.session.run(self._ops.init_op)

        state = types.SimpleNamespace(step=self._step,
                                      train_loss=None,
                                      last_val_loss=None,
                                      best_val_loss=np.inf,
                                      best_val_step=-1)

        def validate_and_save():
            state.last_val_loss = self.validate(write_summaries=True)
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
                state.train_loss, _ = self.training_step()
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

    def training_step(self, feed_dict=None, write_summaries=True, log=True):
        if feed_dict is None:
            feed_dict = {}

        _, train_summary, train_loss = self._dataset_manager.run(
            self.session, (self._ops.train_op, self._ops.summary_op, self._ops.loss),
            self._train_dataset_name, feed_dict={self._ops.training_placeholder: True, **feed_dict})

        self._step = self.session.run(self._global_step_tensor)

        if self._step % self._logging_period == 0:
            if write_summaries and train_summary and self._writer:
                self._writer.add_summary(train_summary, self._step)
            if log:
                logger.info('step: {}, loss: {}'.format(self._step, train_loss))

        return train_loss, train_summary

    def validate(self, write_summaries=False):
        val_losses = self._dataset_manager.run_over_dataset(self.session, self._ops.loss,
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

    def load_variables(self, checkpoint_name='best', checkpoint_file=None):
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
        if self._writer:
            self._writer.add_summary(summary, step if step is not None else self._step)

    class TrainingOps:
        """
        A data class for operations and tensors needed for training.

        Attributes:
            loss: The loss tensor.
            train_op: The training operation.
            init_op: An operation to run before starting training.
            summary_op: A summary to log during training.
            training_placeholder: A boolean placeholder with a default `False` value. The trainer
                will feed this placeholder with True to indicate that the graph should be executed
                in training mode. This is important for techniques like dropout that should only
                be turned on during training.
        """

        def __init__(self, loss, train_op, init_op=(), summary_op=(), training_placeholder=None):  # pylint: disable=unused-argument
            args = vars()
            for name in ['loss', 'train_op', 'init_op', 'summary_op', 'training_placeholder']:
                setattr(self, name, args[name])
