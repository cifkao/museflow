import os

import numpy as np
import tensorflow as tf

from museflow import logger
from museflow.config import configurable


@configurable(['latest_saver', 'best_saver'])
class BasicTrainer:
    """A class implementing a basic training/validation loop, model saving and model loading."""

    def __init__(self, dataset_manager, logdir, logging_period, validation_period, session=None,
                 train_dataset_name='train', val_dataset_name='val'):
        self.session = session or tf.Session()
        self._dataset_manager = dataset_manager
        self._logdir = logdir
        self._logging_period = logging_period
        self._validation_period = validation_period

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._step = 0
        self._train_dataset_name = train_dataset_name
        self._val_dataset_name = val_dataset_name

        self._writer = tf.summary.FileWriter(logdir=self._logdir, graph=tf.get_default_graph())
        with tf.name_scope('savers'):
            self._latest_saver = self._cfg.configure('latest_saver', tf.train.Saver, name='latest')
            self._best_saver = self._cfg.configure('best_saver', tf.train.Saver,
                                                   name='best', max_to_keep=1)

    @property
    def step(self):
        return self._step

    def train(self, train_op, loss, init_op=(), train_summary_op=()):
        self.session.run(init_op)

        best_mean_loss = np.inf

        def validate_and_save(loss):
            nonlocal best_mean_loss

            mean_loss = self.validate(loss)
            if mean_loss < best_mean_loss:
                best_mean_loss = mean_loss
                self.save_variables('best')
            self.save_variables('latest')

        while True:
            if self._step % self._validation_period == 0:
                validate_and_save(loss)

            try:
                _, train_summary, train_loss = self._dataset_manager.run(
                    self.session, (train_op, train_summary_op, loss), self._train_dataset_name,
                    feed_dict={self._dataset_manager.training: True})
            except tf.errors.OutOfRangeError:
                break

            self._step = self.session.run(self._global_step_tensor)

            if self._step % self._logging_period == 0:
                if train_summary:
                    self._writer.add_summary(train_summary, self._step)
                logger.info('step: {}, loss: {}'.format(self._step, train_loss))

            if np.isnan(train_loss):
                logger.error('NaN loss, stopping training')
                break

        validate_and_save(loss)

    def validate(self, loss, write_summaries=True):
        val_losses = self._dataset_manager.run_over_dataset(self.session, loss,
                                                            self._val_dataset_name)
        mean_loss = np.mean(val_losses)

        if write_summaries:
            val_summary = tf.Summary(value=[
                tf.Summary.Value(tag='{}/loss'.format(self._val_dataset_name),
                                 simple_value=mean_loss)
            ])
            self._writer.add_summary(val_summary, self._step)

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
