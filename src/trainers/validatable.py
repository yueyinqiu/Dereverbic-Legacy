import abc
from typing import Any

from statictorch import Tensor2d


class Validatable(abc.ABC):
    @abc.abstractmethod
    def validate_on(self, 
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        ...
