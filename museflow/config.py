"""Model configuration.

This module implements a configuration mechanism, which enables a model to be created from a
configuration dictionary, e.g.:

```
config_dict = {
    'say_hello': True,
    'encoder': {
        'cell': {
            'num_units': 500
        }
    },
    'decoder': {
        'size': 1000
    }
}
configure(MyModel, config_dict, logdir='/tmp/logdir')
```

To make our model class configurable in the first place, we need to decorate it with
`@configurable`. When the decorated class is instantiated, a magic `_cfg` attribute is attached to
it automatically. This `_cfg` is an instance of the `Configuration` class and can be used to
create other configurable objects inside the model. As an example:

```
@configurable(['encoder', 'decoder'])
class MyModel(Model):

    def __init__(self, logdir, say_hello=False):
        Model.__init__(self, logdir)

        if say_hello:
            logger.info('Hello world!')
        encoder = self._cfg.configure('encoder', MyEncoder)
        decoder = self._cfg.configure('decoder', MyDecoder)
```
"""

import functools
import sys

from museflow import logger


class Configuration:

    def __init__(self, config_dict, configurables=()):
        self._config_dict = config_dict
        self._subconfigs = configurables

    def configure(self, *args, **kwargs):
        """Configure an object using a given key from the config dict.

        Two positional arguments are expected: `config_key` (required) and `constructor` (optional).
        Calls `constructor` with the keyword arguments specified in `config_dict[config_key]` or
        passed to this function. Note that the constructor is called even if `config_key` is not
        present in `config_dict`.

        Any keyword arguments passed to this function are treated as defaults and can be overridden
        by the config dict.

        Returns:
            The return value of `constructor`, or `None` if `config[config_key]` is `None`.
        """
        config_key, constructor = _expand_args(self.configure, args, 1, None)

        if config_key not in self._subconfigs:
            raise RuntimeError('Key {} not defined in {}._subconfigs'.format(
                config_key, type(self).__name__))

        config = self._config_dict.get(config_key, {})

        if config is None:
            return None

        if type(config) is not dict:
            if constructor or kwargs:
                raise ConfigError('Error while configuring {}: dict expected, got {}'.format(
                    config_key, type(config)
                ))
            return config

        try:
            config_dict = dict(config)  # Make a copy of the dict
            if not constructor or 'class' in config_dict:
                constructor = config_dict['class']
                del config_dict['class']
                if not constructor:
                    return None
        except Exception as e:
            raise ConfigError('{} while configuring {}: {}'.format(
                type(e).__name__, config_key, e
            )).with_traceback(sys.exc_info()[2]) from None

        try:
            if hasattr(constructor, '__museflow_subconfigs'):
                return _construct_configurable(constructor, kwargs, config_dict)

            _log_call(constructor, kwargs={**kwargs, **config_dict})
            return constructor(**kwargs, **config_dict)
        except TypeError as e:
            raise ConfigError('{} while configuring {} ({!r}): {}'.format(
                type(e).__name__, config_key, constructor, e
            )).with_traceback(sys.exc_info()[2]) from None

    def maybe_configure(self, *args, **kwargs):
        """Configure an object using a given key only if the key is present in the config dict.

        Like `_configure`, but returns `None` if the key is not present.
        """
        config_key, constructor = _expand_args(self.maybe_configure, args, 1, None)

        if config_key not in self._subconfigs:
            raise RuntimeError('Key {} not defined in {}._subconfigs'.format(
                config_key, type(self).__name__))

        if config_key not in self._config_dict:
            return None

        return self.configure(config_key, constructor, **kwargs)


def configurable(subconfigs=()):
    """Return a decorator that makes a function or a class configurable.

    The configurable function/class can be called/instantiated normally, or via `configure` or
    `Configuration.configure`.

    A configurable function should have an extra first argument `cfg`, which will be automatically
    populated with an instance of `Configuration` when the function is called. It is then possible
    to call `cfg.configure` or `cfg.maybe_configure` in the body of the function.

    A configurable class can access its `Configuration` object via `self._cfg` (e.g.
    `self._cfg.configure`).

    Args:
        subconfigs: A list of names of objects that can be configured via the configuration
            mechanism.
    Returns:
        The decorator.
    """
    def decorator(x):
        parent_subconfigs = getattr(x, '__museflow_subconfigs', ())
        setattr(x, '__museflow_subconfigs', (*parent_subconfigs, *subconfigs))

        # Wrap x or its __init__ so that an empty Configuration gets created by default.

        if isinstance(x, type):
            init = x.__init__

            @functools.wraps(init)
            def init_wrapper(self, *args, **kwargs):
                if not hasattr(self, '_cfg'):
                    cfg = Configuration({}, getattr(self, '__museflow_subconfigs'))
                    setattr(self, '_cfg', cfg)

                init(self, *args, **kwargs)

            x.__init__ = init_wrapper
            return x
        else:
            @functools.wraps(x)
            def wrapper(*args, **kwargs):
                cfg = Configuration({}, getattr(x, '__museflow_subconfigs'))
                return x(cfg, *args, **kwargs)

            setattr(wrapper, '__museflow_wrapped', x)
            return wrapper

    return decorator


def configure(*args, **kwargs):
    """Call/instantiate a configurable function/class from a config dict.

    Two positional arguments are required: the function or class to be configured and the config
    dict.
    """
    x, config_dict = _expand_args(configure, args, 2)
    return _construct_configurable(x, kwargs, config_dict)


def _construct_configurable(x, kwargs, config_dict):
    subconfigs = getattr(x, '__museflow_subconfigs')

    # Move keys from config_dict to kwargs, except for those that are in the subconfigs list.
    kwargs = dict(kwargs)
    kwargs.update({k: v for k, v in config_dict.items() if k not in subconfigs})
    config_dict = {k: v for k, v in config_dict.items() if k in subconfigs}
    _log_call(x, kwargs=kwargs)

    cfg = Configuration(config_dict, subconfigs)

    if isinstance(x, type):
        obj = x.__new__(x, **kwargs)
        setattr(obj, '_cfg', cfg)
        obj.__init__(**kwargs)
        return obj
    else:
        wrapped = getattr(x, '__museflow_wrapped')
        return wrapped(cfg, **kwargs)


def _expand_args(fn, args, required, *defaults):
    if len(args) < required:
        raise TypeError('{} expected at least {} positional arguments, got {}'.format(
            fn.__name__, required, len(args)))

    max_num = required + len(defaults)
    if len(args) > max_num:
        raise TypeError('{} expected at most {} positional arguments, got {}'.format(
            fn.__name__, max_num, len(args)))

    return args + defaults[len(args) - required:]


def _log_call(fn, args=None, kwargs=None):
    args = args or []
    kwargs = kwargs or {}
    args_and_kwargs = [f'{a!r}' for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
    logger.debug('Calling {}({})'.format(fn.__name__, ', '.join(args_and_kwargs)))


class ConfigError(Exception):
    pass
