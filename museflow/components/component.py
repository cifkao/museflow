from contextlib import contextmanager
from functools import wraps

from cached_property import cached_property
import tensorflow as tf


def using_scope(function):
    def wrapper(self, *args, **kwargs):
        with self.use_scope():
            return function(self, *args, **kwargs)

    return wrapper


class Component:

    def __init__(self, name):
        self.name = name
        self._built = False

        # Create a variable scope for this model part
        with tf.variable_scope(name) as scope:
            self.variable_scope = scope

    @contextmanager
    def use_scope(self, reuse=False):
        # Make sure we use the original variable and name scope
        with tf.variable_scope(self.variable_scope, reuse=reuse):
            with tf.name_scope(self.variable_scope.original_name_scope):
                yield

    @cached_property
    def trainable_variables(self):
        if not self.built:
            raise RuntimeError("Attempt to access 'trainable_variables' before model is built")
        return tf.trainable_variables(self.variable_scope.name)

    @property
    def built(self):
        """Indicates whether all variables have been created."""
        return self._built
