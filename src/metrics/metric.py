import abc
from typing import Generic, TypeVar


_T = TypeVar("_T", contravariant=True)   # pylint: disable=un-declared-variable

class Metric(Generic[_T], abc.ABC):
    @abc.abstractmethod
    def append(self, actual: _T, predicted: _T) -> dict[str, float]:
        ...
    
    @abc.abstractmethod
    def result(self) -> dict[str, float]:
        ...
