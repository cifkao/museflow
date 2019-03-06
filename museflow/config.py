"""Model configuration.

This module implements a configuration mechanism, which enables a model to be created from a
configuration dictionary, e.g.:

```
config_dict = {
    'encoder': {
        'cell1': {
            'num_units': 500
        },
        'name': 'encoder'
    },
    'say_hello': True,
}
config = Configuration(config_dict)
config.configure(MyModel, logdir='/tmp/logdir')
```

To make our model class configurable in the first place, we need to decorate it with
`@configurable`. When the decorated model class is instantiated, a magic `_cfg` attribute is
attached to the instance automatically. This `_cfg` is the `Configuration` object that was used to
configure the model, and can be used to configure other objects inside the model.
As an example:

```
@configurable(['encoder'])
class MyModel:

    def __init__(self, logdir, say_hello=False):
        if say_hello:
            logger.info('Hello world!')
        encoder = self._cfg['encoder'].configure(MyEncoder)

@configurable(['cell1', 'cell2'])
class MyEncoder:

    def __init__(self, name='my_encoder'):
        cell1 = self._cfg['cell1'].configure(LSTMCell, num_units=50)  # overriden with num_units=500
        cell2 = self._cfg['cell2'].configure(LSTMCell, num_units=100)

```

In this example, `MyModel` will be called with the parameters `logdir='/tmp/logdir'` and
`say_hello=True`. Inside the model, we then instantiate `MyEncoder(name='encoder')`, which in turn
creates two instances of `LSTMCell` with `num_units=500` and `num_units=100`, respectively.

Note that:
  - Keyword arguments passed to `configure` can be overridden in the configuration dict.
  - Inside `MyEncoder`, we can call `self._cfg['cell2'].configure` even though the key `'cell2'` is
    not defined in the configuration.
  - We need to pass `['encoder']` to the decorator for `MyModel`, otherwise the corresponding value
    from the configuration dict would get passed as a keyword argument.
"""

import functools
import sys

import yaml

from museflow import logger


_MISSING_VALUE = object()
_NO_DEFAULT = object()


class Configuration:

    def __init__(self, value, name='<default>'):
        self._wrapped = value
        self.name = name
        self._child_configs = {}

        self._is_special_name = name.startswith('<')

    def get(self, key=None, default=_NO_DEFAULT):
        """Return an item from this configuration object.

        Returns:
            If `key` is given, the corresponding item from the wrapped object. Otherwise, the entire
            wrapped value. If the key is missing, `default` is returned instead (if given).
        Raises:
            KeyError: If the key is missing and no default was given.
            IndexError: If the key is missing and no default was given.
            TypeError: If the wrapped object does not support indexing.
        """
        if self._wrapped is _MISSING_VALUE:
            if default is _NO_DEFAULT:
                raise KeyError(f'Missing configuration value {self._name_repr}')
            return default

        if key is None:
            return self._wrapped

        if not hasattr(self._wrapped, '__getitem__'):
            raise TypeError(f'Attempted to get item {repr(key)} of configuration object '
                            f'{self._name_repr} of type {type(self._wrapped)}')

        try:
            return self._wrapped[key]
        except (KeyError, IndexError) as e:
            if default is _NO_DEFAULT:
                raise type(e)(f"Missing configuration value '{self._get_key_name(key)}'") from None
            return default

    def configure(self, *args, **kwargs):
        """Configure an object using this configuration.

        One optional positional argument is expected: `constructor`.
        Calls `constructor` with the keyword arguments specified in this configuration object or
        passed to this function. Note that the constructor is called even if this configuration
        object corresponds to a missing key.

        Any keyword arguments passed to this function are treated as defaults and can be overridden
        by the configuration.

        Returns:
            The return value of `constructor`, or `None` if the value of this configuration object
            is `None`.
        """
        if len(args) > 1:
            raise TypeError(f'Expected at most 1 positional argument, got {len(args)}')
        constructor = args[0] if args else None

        config_val = self.get(default={})  # May raise TypeError
        if config_val is None:
            return None

        # If the value is a list, we treat each item separately.
        # We create a Configuration object for each item by accessing self[i].
        if type(config_val) is list:
            return [self._configure(self[i], config_item, constructor, kwargs)
                    for i, config_item in enumerate(config_val)]

        return self._configure(self, config_val, constructor, kwargs)

    def maybe_configure(self, *args, **kwargs):
        """Configure an object only if a configuration is present.

        Like `configure`, but returns `None` if the configuration is missing.
        """
        if len(args) > 1:
            raise TypeError(f'Expected at most 1 positional argument, got {len(args)}')
        constructor = args[0] if args else None

        if self._wrapped is _MISSING_VALUE:
            return None

        return self.configure(constructor, **kwargs)

    def _configure(self, config, config_val, constructor, kwargs):
        if type(config_val) is not dict:
            if constructor or kwargs:
                raise ConfigError(f'Error while configuring {self._name_repr}: dict expected, '
                                  f'got {type(config_val)}')
            return config_val
        config_dict = dict(config_val)  # Make a copy of the dict

        try:
            if not constructor or 'class' in config_dict:
                try:
                    constructor = config_dict['class']
                except KeyError:
                    raise ConfigError('No constructor (class) specified') from None
                del config_dict['class']
                if not constructor:
                    return None
        except Exception as e:
            raise ConfigError('{} while configuring {}: {}'.format(
                type(e).__name__, self._name_repr, e
            )).with_traceback(sys.exc_info()[2]) from None

        # If the constructor is decorated with @configurable, we use _construct_configurable, which
        # creates a Configuration object and passes it to the constructor. Otherwise, we just call
        # the constructor.
        try:
            if hasattr(constructor, '__museflow_subconfigs'):
                return _construct_configurable(constructor, kwargs, config_dict, cfg=config)

            kwargs = {**kwargs, **config_dict}
            _log_call(constructor, kwargs=kwargs)
            return constructor(**kwargs)
        except TypeError as e:
            raise ConfigError('{} while configuring {} ({!r}): {}'.format(
                type(e).__name__, self._name_repr, constructor, e
            )).with_traceback(sys.exc_info()[2]) from None

    def __getitem__(self, key):
        """Return a `Configuration` object corresponding to the given key.

        If the key is missing in the wrapped object, a `Configuration` object with a special
        `<missing>` value is returned.
        """
        if key not in self._child_configs:
            self._child_configs[key] = Configuration(self.get(key, _MISSING_VALUE),
                                                     name=self._get_key_name(key))
        return self._child_configs[key]

    def __setitem__(self, key, value):
        try:
            self._wrapped[key] = value
        except TypeError as e:
            raise TypeError(f'{self.name}: {e}') from None
        if key in self._child_configs:
            setattr(self._child_configs[key], '_wrapped', value)

    def __delitem__(self, key):
        try:
            del self._wrapped[key]
        except (KeyError, IndexError, TypeError) as e:
            raise type(e)(f'{self.name}: {e}') from None

    def __iter__(self):
        try:
            return iter(self._wrapped)
        except TypeError as e:
            raise TypeError(f'{self.name}: {e}') from None

    def __len__(self):
        try:
            return len(self._wrapped)
        except TypeError as e:
            raise TypeError(f'{self.name}: {e}') from None

    def __contains__(self, key):
        try:
            return key in self._wrapped
        except TypeError as e:
            raise TypeError(f'{self.name}: {e}') from None

    def __bool__(self):
        return self._wrapped is not _MISSING_VALUE and bool(self._wrapped)

    def __repr__(self):
        return 'Configuration({}{})'.format(
            repr(self._wrapped) if self._wrapped is not _MISSING_VALUE else '<missing>',
            f', name={self._name_repr}' if not self._is_special_name else '')

    @property
    def _name_repr(self):
        return repr(self.name) if not self._is_special_name else self.name

    def _get_key_name(self, key):
        if not self._is_special_name:
            return f'{self.name}.{key}' if isinstance(key, str) else f'{self.name}[{key}]'
        return key

    @classmethod
    def from_yaml(cls, stream):
        return cls(yaml.load(stream), '<root>')


def configurable(subconfigs=(), pass_kwargs=True):
    """Return a decorator that makes a function or a class configurable.

    The configurable function/class can be called/instantiated normally, or via
    `Configuration.configure`.

    A configurable function should have an extra first argument `cfg`, which will be automatically
    populated with an instance of `Configuration` when the function is called. A configurable class
    can access its `Configuration` object via `self._cfg`.

    Args:
        subconfigs: A list of configuration items that are to be used via the configuration
            mechanism, and therefore should not be passed as keyword arguments to the configurable.
        pass_kwargs: If `True` (default), all configuration items not specified in `child_configs`
            will be passed as keyword arguments when calling the configurable.
    Returns:
        The decorator.
    """
    def decorator(x):
        parent_subconfigs = getattr(x, '__museflow_subconfigs', ())
        setattr(x, '__museflow_subconfigs', (*parent_subconfigs, *subconfigs))
        setattr(x, '__museflow_pass_kwargs', pass_kwargs)

        # Wrap x or its __init__ so that an empty Configuration gets created by default.

        if isinstance(x, type):
            init = x.__init__

            @functools.wraps(init)
            def init_wrapper(self, *args, **kwargs):
                if not hasattr(self, '_cfg'):
                    cfg = Configuration(_MISSING_VALUE)
                    setattr(self, '_cfg', cfg)

                init(self, *args, **kwargs)

            x.__init__ = init_wrapper
            return x
        else:
            @functools.wraps(x)
            def wrapper(*args, **kwargs):
                cfg = Configuration(_MISSING_VALUE)
                return x(cfg, *args, **kwargs)

            setattr(wrapper, '__museflow_wrapped', x)
            return wrapper

    return decorator


def _construct_configurable(x, kwargs, config_dict, cfg):
    subconfigs = getattr(x, '__museflow_subconfigs')
    pass_kwargs = getattr(x, '__museflow_pass_kwargs')

    if pass_kwargs:
        # Copy keys from config_dict to kwargs, except for those that are in the subconfigs list.
        kwargs = dict(kwargs)
        kwargs.update({k: v for k, v in config_dict.items() if k not in subconfigs})

    _log_call(x, kwargs=kwargs)

    if isinstance(x, type):
        obj = x.__new__(x, **kwargs)
        setattr(obj, '_cfg', cfg)
        obj.__init__(**kwargs)
        return obj
    else:
        wrapped = getattr(x, '__museflow_wrapped')
        return wrapped(cfg, **kwargs)


def _log_call(fn, args=None, kwargs=None):
    args = args or []
    kwargs = kwargs or {}
    args_and_kwargs = [f'{a!r}' for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
    logger.debug('Calling {}({})'.format(fn.__name__, ', '.join(args_and_kwargs)))


class ConfigError(Exception):
    pass
