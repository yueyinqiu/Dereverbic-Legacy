from .i0 import *


class RirBlindEstimationModel(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @abc.abstractmethod
    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        ...
