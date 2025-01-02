from .imports import *


class RirBlindEstimationModel(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @abc.abstractmethod
    def train_on(self, reverb_batch: Tensor, rir_batch: Tensor, speech_batch: Tensor) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def evaluate_on(self, reverb_batch: Tensor) -> Tensor:
        ...