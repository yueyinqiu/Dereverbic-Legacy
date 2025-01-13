from .imports import *
from .dimension_descriptors import *


class RirBlindEstimationModel(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @abc.abstractmethod
    def train_on(self, 
                 reverb_batch: Tensor2d[DBatch, DSample], 
                 rir_batch: Tensor2d[DBatch, DSample], 
                 speech_batch: Tensor2d[DBatch, DSample]) -> dict[str, float]:
        ...
