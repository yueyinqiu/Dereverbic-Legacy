import abc

from statictorch import Tensor2d


class Metric(abc.ABC):
    @abc.abstractmethod
    def append(self, actual: Tensor2d, predicted: Tensor2d) -> dict[str, float]:
        ...
    
    @abc.abstractmethod
    def result(self) -> dict[str, float]:
        ...
