from statictorch import Tensor1d, Tensor2d
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from metrics.metric import Metric


class BiasMetric(Metric[Tensor1d | Tensor2d]):
    def __init__(self):
        self._accumulator = KahanAccumulator()
        self._count = 0
    
    def append(self, actual: Tensor1d | Tensor2d, predicted: Tensor1d | Tensor2d) -> dict[str, float]:
        value: float = float(torch.mean(predicted - actual))

        self._accumulator.add(value)
        self._count += 1
        
        return {
            "value": value
        }
        
    def result(self) -> dict[str, float]:
        return {
            "value": self._accumulator.value() / self._count
        }
