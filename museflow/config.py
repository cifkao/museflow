import sys

from museflow import logger


_NO_DEFAULT = object()


class Configurable:
    _subconfigs = []

    def __init__(self, config=None):
        self._config_dict = config or {}

    def _configure(self, config_key, constructor=None, **kwargs):
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
            # If it's a class which is a subclass of Configurable...
            if isinstance(constructor, type) and issubclass(constructor, Configurable):
                return constructor.from_config(config_dict, **kwargs)

            _log_call(constructor, **kwargs, **config_dict)
            return constructor(**kwargs, **config_dict)
        except TypeError as e:
            raise ConfigError('{} while configuring {} ({!r}): {}'.format(
                type(e).__name__, config_key, constructor, e
            )).with_traceback(sys.exc_info()[2]) from None

    def _maybe_configure(self, config_key, constructor=None, **kwargs):
        return self._configure(config_key, constructor, _default=None, **kwargs)

    def _get_config(self, key, default=_NO_DEFAULT):
        if default is _NO_DEFAULT:
            return self._config_dict[key]
        return self._config_dict.get(key, default)

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update({k: v for k, v in config.items() if k not in cls._subconfigs})
        config = {k: v for k, v in config.items() if k in cls._subconfigs}

        _log_call(cls, *args, **kwargs)
        return cls(*args, **kwargs, config=config)


def _log_call(fn, *args, **kwargs):
    args_and_kwargs = [f'{a}' for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
    logger.debug('Calling {}({})'.format(fn.__name__, ', '.join(args_and_kwargs)))


class ConfigError(Exception):
    pass
