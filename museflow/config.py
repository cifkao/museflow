import functools
import sys

from museflow import logger


_NO_DEFAULT = object()


class Configurable:
    """A base class for configurable objects.

    Every subclass of `Configurable` should accept a `config` argument in its `__init__` method and
    pass it to `Configurable`. The subclass may also define the `_subconfigs` class attribute to
    list objects that can be configured via this config dict. An example:

    ```
    class MyModel(Configurable):
        _subconfigs = ['encoder', 'decoder']

        def __init__(self, config=None):
            Configurable.__init__(self, config)

            encoder = self._configure('encoder', MyEncoder)
            decoder = self._configure('decoder', MyDecoder)
    ```

    In this example, calling `self._configure('encoder', MyEncoder)` will construct an instance of
    the `MyEncoder` class with the keyword arguments given in `config['encoder']`.
    """

    _subconfigs = []

    def __init__(self, config=None):
        self._config_dict = config or {}

    def _configure(self, config_key, constructor=None, **kwargs):
        """Configure an object using a given key from the config dict.

        Calls `constructor` with the keyword arguments specified in `config[config_key]`. Note that
        the constructor is called even if `config_key` is not present in `config`.

        Args:
            config_key: A key specified in `_subconfigs`.
            constructor: A callable that constructs the object. Can be overridden by a `class` key
                in the config dict.
            **kwargs: Default keyword arguments to pass to the constructor. Can be overridden by
                values in the config dict.
        Returns:
            The return value of `constructor`, or `None` if `config[config_key]` is `None`.
        """
        if config_key not in self._subconfigs:
            raise RuntimeError('Key {} not defined in {}._subconfigs'.format(
                config_key, type(self).__name__))

        _default = {}
        if '_default' in kwargs:
            _default = kwargs['_default']
            del kwargs['_default']
        config = self._get_config(config_key, _default)

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
            # If it's a class which is a subclass of Configurable or if it's a configurable function
            if ((isinstance(constructor, type) and issubclass(constructor, Configurable)) or
                    isinstance(constructor, _ConfigurableFunction)):
                return constructor.from_config(config_dict, **kwargs)

            _log_call(constructor, **kwargs, **config_dict)
            return constructor(**kwargs, **config_dict)
        except TypeError as e:
            raise ConfigError('{} while configuring {} ({!r}): {}'.format(
                type(e).__name__, config_key, constructor, e
            )).with_traceback(sys.exc_info()[2]) from None

    def _maybe_configure(self, config_key, constructor=None, **kwargs):
        """Configure an object using a given key only if the key is present in the config dict.

        Like `_configure`, but returns `None` if the key is not present.
        """
        return self._configure(config_key, constructor, _default=None, **kwargs)

    def _get_config(self, key, default=_NO_DEFAULT):
        """Retrieve a given key from the config dict."""
        if default is _NO_DEFAULT:
            return self._config_dict[key]
        return self._config_dict.get(key, default)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        """Construct an instance of this class from a given config dict.

        Args:
            config: A config dict containing keyword arguments for the `__init__` method and keys
                listed in `_subconfigs`.
            *args: Positional arguments to the `__init__` method.
            **kwargs: Default keyword arguments to the `__init__` method. Can be overridden by
                values in `config`.
        Returns:
            The created instance.
        """
        kwargs = dict(kwargs)
        kwargs.update({k: v for k, v in config.items() if k not in cls._subconfigs})
        config = {k: v for k, v in config.items() if k in cls._subconfigs}

        _log_call(cls, *args, **kwargs)
        return cls(*args, **kwargs, config=config)


def configurable(subconfigs=None):
    """Return a decorator that makes a function configurable.

    The wrapped (decorated) function should have an extra first argument `cfg`. It is then possible
    to call `cfg.configure` or `cfg.maybe_configure` in the body of the function. The function can
    be used in the same way as a configurable type or called normally (without the `cfg` argument).

    Args:
        subconfigs: A list of objects that can be configured via the configuration mechanism.
    Returns:
        The decorator.
    """
    return functools.partial(_ConfigurableFunction, subconfigs=subconfigs)


class _ConfigurableFunction:

    def __init__(self, function, subconfigs):
        self._function = function
        functools.update_wrapper(self, self._function)

        # We need to create a new Configurable type so that we can set its _subconfigs
        self._configurator = type(self._function.__name__ + '__cfg',
                                  (self.Configurator,),
                                  dict(_subconfigs=subconfigs))

    def __call__(self, *args, **kwargs):
        return self._function(self._configurator(), *args, **kwargs)

    def from_config(self, config, *args, **kwargs):
        cfg = self._configurator.from_config(config, *args, **kwargs)
        _log_call(self._function, *args, **kwargs)
        return self._function(cfg, *cfg.fn_args, **cfg.fn_kwargs)

    class Configurator(Configurable):

        def __init__(self, *args, config=None, **kwargs):
            Configurable.__init__(self, config)
            self.fn_args = args
            self.fn_kwargs = kwargs
            self.configure = self._configure
            self.maybe_configure = self._maybe_configure


def _log_call(fn, *args, **kwargs):
    if isinstance(fn, type) and issubclass(fn, _ConfigurableFunction.Configurator):
        return
    args_and_kwargs = [f'{a}' for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
    logger.debug('Calling {}({})'.format(fn.__name__, ', '.join(args_and_kwargs)))


class ConfigError(Exception):
    pass
