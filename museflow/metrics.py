from abc import ABCMeta, abstractproperty, abstractmethod


class Metric(metaclass=ABCMeta):

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def fetches(self):
        pass

    @abstractmethod
    def compute(self, results):
        pass
