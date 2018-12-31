from contextlib import contextmanager

from cached_property import threaded_cached_property as cached_property
import tensorflow as tf


def using_scope(function):
    """A decorator that wraps a method in `self.use_scope()`."""
    def wrapper(self, *args, **kwargs):
        with self.use_scope():
            return function(self, *args, **kwargs)

    return wrapper


class Component:
    """A model component."""

    def __init__(self, name):
        """Initialize the component.

        Args:
            name: A name for the component's name scope and variable scope.
        """
        self.name = name
        self._built = False

        # Create a variable scope for this component
        with tf.variable_scope(name) as scope:
            self.variable_scope = scope

    @contextmanager
    def use_scope(self, reuse=False):
        """Get a context manager that opens the component's name scope and variable scope."""
        # Make sure we use the original variable and name scope
        with tf.variable_scope(self.variable_scope, reuse=reuse):
            with tf.name_scope(self.variable_scope.original_name_scope):
                yield

    @cached_property
    def trainable_variables(self):
        """A list of trainable variables in the component's variable scope."""
        if not self.built:
            raise RuntimeError("Attempt to access 'trainable_variables' before model is built")
        return tf.trainable_variables(self.variable_scope.name)

    @property
    def built(self):
        """Indicates whether all variables have been created."""
        return self._built
