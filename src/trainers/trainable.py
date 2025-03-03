import abc
from typing import Any

from statictorch import Tensor2d


class Trainable(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @abc.abstractmethod
    def prepare_train_on(self, 
                         reverb_batch: Tensor2d, 
                         rir_batch: Tensor2d, 
                         speech_batch: Tensor2d) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def train_prepared(self):
        ...

    @abc.abstractmethod
    def validate_on(self, 
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        ...
