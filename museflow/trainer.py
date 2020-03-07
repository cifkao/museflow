import abc
import copy
import os
import types

from confugue import configurable
import numpy as np
import tensorflow as tf

from museflow import logger


class BaseTrainer(metaclass=abc.ABCMeta):
    """A base class for trainers, implementing a training step and model loading and saving."""

    def __init__(self, dataset_manager, logdir, logging_period, training_ops=None,
                 session=None, write_summaries=True):
        self.session = session or tf.Session()
        self._dataset_manager = dataset_manager
        self._logdir = logdir
        self._logging_period = logging_period
        self._ops = training_ops or BasicTrainer.TrainingOps(loss=None, train_op=())

        self._global_step_tensor = tf.train.get_or_create_global_step()
        self._global_step_initialized = tf.is_variable_initialized(self._global_step_tensor)
        self._step = 0

        if self._ops.training_placeholder is None:
            self._ops.training_placeholder = tf.placeholder_with_default(False, [],
                                                                         name='is_training')

        if write_summaries:
            self._writer = tf.summary.FileWriter(logdir=self._logdir, graph=tf.get_default_graph())
        else:
            self._writer = None

    @property
    def step(self):
        return self._step

    def training_step(self, dataset_name, feed_dict=None, write_summaries=True, log=True):
        if feed_dict is None:
            feed_dict = {}

        _, train_summary, train_loss = self._dataset_manager.run(
            self.session, (self._ops.train_op, self._ops.summary_op, self._ops.loss),
            dataset_name, feed_dict={self._ops.training_placeholder: True, **feed_dict})

        self._step = self.session.run(self._global_step_tensor)

        if self._step % self._logging_period == 0:
            if write_summaries and train_summary and self._writer:
                self._writer.add_summary(train_summary, self._step)
            if log:
                logger.info('step: {}, loss: {}'.format(self._step, train_loss))

        return train_loss, train_summary

    def save_variables(self, checkpoint_name):
        self._get_saver(checkpoint_name).save(
            self.session,
            os.path.join(self._logdir, checkpoint_name + '.ckpt'),
            latest_filename=checkpoint_name + '_checkpoint',
            global_step=self._step)

    def load_variables(self, checkpoint_name, checkpoint_file=None):
        if not checkpoint_file:
            checkpoint_file = tf.train.latest_checkpoint(self._logdir,
                                                         checkpoint_name + '_checkpoint')
        self._get_saver(checkpoint_name).restore(self.session, checkpoint_file)
        logger.info('Variables restored from {}'.format(checkpoint_file))
        if self.session.run(self._global_step_initialized):
            self._step = self.session.run(self._global_step_tensor)

    def write_scalar_summary(self, name, value, step=None):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=value)
        ])
        if self._writer:
            self._writer.add_summary(summary, step if step is not None else self._step)

    @abc.abstractmethod
    def _get_saver(self, checkpoint_name):
        """Returns the saver corresponding to the given checkpoint name."""

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


DEFAULT = object()


@configurable
class BasicTrainer(BaseTrainer):
    """A trainer implementing a basic training/validation loop."""

    def __init__(self, dataset_manager, logdir, logging_period, validation_period=None,
                 training_ops=None, session=None, write_summaries=True,
                 train_dataset_name='train', val_dataset_name='val'):
        super().__init__(dataset_manager=dataset_manager,
                         logdir=logdir,
                         logging_period=logging_period,
                         training_ops=training_ops,
                         session=session,
                         write_summaries=write_summaries)

        self._validation_period = validation_period
        self._train_dataset_name = train_dataset_name
        self._val_dataset_name = val_dataset_name

        self._savers = {
            'latest': self._cfg['latest_saver'].configure(tf.train.Saver,
                                                          name='latest', max_to_keep=1),
            'best': self._cfg['best_saver'].configure(tf.train.Saver,
                                                      name='best', max_to_keep=1)
        }

    def train(self, dataset_name=None):
        train_generator = self.iter_train(period=None, dataset_name=dataset_name)
        return next(train_generator)

    def iter_train(self, dataset_name=None, period=DEFAULT):
        """Return a generator that runs the training loop.

        Every `period` training steps, the generator yields an object with the attributes `step`,
        `train_loss`, `last_val_loss`, `best_val_loss` and `best_val_step`. Note that the first
        time, `step` will be 0 and the loss values possibly unknown.

        If `period` is not specified, `validation_period` is used instead.
        """
        if period == DEFAULT:
            period = self._validation_period

        def training_step_fn():
            loss, _ = self.training_step(dataset_name=dataset_name or self._train_dataset_name)
            return self._step, loss

        def validation_fn():
            self.save_variables('latest')
            return self.validate(write_summaries=True)

        return training_validation_loop(training_step_fn=training_step_fn,
                                        validation_fn=validation_fn,
                                        init_fn=lambda: self.session.run(self._ops.init_op),
                                        best_val_loss_fn=lambda _: self.save_variables('best'),
                                        validation_period=self._validation_period,
                                        yield_period=period,
                                        initial_step=self._step)

    def validate(self, write_summaries=False):
        val_losses = self._dataset_manager.run_over_dataset(self.session, self._ops.loss,
                                                            self._val_dataset_name)
        mean_loss = np.mean(val_losses) if val_losses else np.nan

        if write_summaries:
            self.write_scalar_summary('{}/loss'.format(self._val_dataset_name), mean_loss)

        return mean_loss

    def save_variables(self, checkpoint_name='latest'):
        super().save_variables(checkpoint_name=checkpoint_name)

    def load_variables(self, checkpoint_name='best', checkpoint_file=None):
        super().load_variables(checkpoint_name=checkpoint_name, checkpoint_file=checkpoint_file)

    def _get_saver(self, checkpoint_name):
        return self._savers[checkpoint_name]


def training_validation_loop(training_step_fn, init_fn=None, validation_fn=None,
                             best_val_loss_fn=None, validation_period=None, yield_period=None,
                             initial_step=0):
    """Return a generator that runs a basic training loop.

    Every `period` training steps, the generator yields an object holding information about the
    state of the training.

    Args:
        training_step_fn: A function that performs one training step and returns the step number
            and the training loss value.
        init_fn: A function to call at the beginning of the training loop.
        validation_fn: A function that performs validation and returns the validation loss.
        best_val_loss_fn: A function to call when the best validation loss is achieved; the value
            of the loss is passed as its only argument.
        validation_period: How many training steps to perform between running the validation.
        yield_period: How many training steps to perform between two yields of the generator.
        initial_step: The initial step value.

    Yields:
        An object with the attributes `step`, `train_loss`, `last_val_loss`, `best_val_loss` and
        `best_val_step`.
    """
    if (validation_period is None) != (validation_fn is None):
        raise ValueError("'validation_period' and 'validation_fn' must be either both None "
                         'or both non-None')

    if init_fn is not None:
        init_fn()

    state = types.SimpleNamespace(step=initial_step,
                                  train_loss=None,
                                  last_val_loss=None,
                                  best_val_loss=np.inf,
                                  best_val_step=-1)

    def validate():
        if validation_fn is None:
            return
        state.last_val_loss = validation_fn()
        if state.last_val_loss < state.best_val_loss:
            state.best_val_loss, state.best_val_step = state.last_val_loss, state.step
            if best_val_loss_fn is not None:
                best_val_loss_fn(state.best_val_loss)

    while True:
        if validation_period is not None and state.step % validation_period == 0:
            validate()
        if yield_period is not None and state.step % yield_period == 0:
            yield copy.copy(state)

        try:
            state.step, state.train_loss = training_step_fn()
        except tf.errors.OutOfRangeError:
            logger.info(f'Training finished after {state.step} steps, loss: {state.train_loss}')
            break

        if np.isnan(state.train_loss):
            logger.error('NaN loss, stopping training')
            break

    if validation_period is not None:
        validate()
    yield state
