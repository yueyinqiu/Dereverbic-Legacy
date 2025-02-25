from typing import NamedTuple
from statictorch import Tensor0d, Tensor2d
from torch import Tensor
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from metrics.metric import Metric


class SisnrMetric(Metric):
    def __init__(self):
        self._accumulator = KahanAccumulator()
        self._count = 0
    
    def _loss(self, actual: Tensor2d, predicted: Tensor2d):
        dot: Tensor = torch.sum(actual * predicted, dim=-1, keepdim=True)
        actual_power: Tensor = torch.sum(actual ** 2, dim=-1, keepdim=True)
        target: Tensor = dot * actual / actual_power

        noise: Tensor = predicted - target

        target_power: Tensor = torch.sum(target ** 2, dim=-1)
        noise_power: Tensor = torch.sum(noise ** 2, dim=-1)
        
        result: Tensor = 10 * torch.log10(target_power / noise_power)
        return torch.mean(result)

    def append(self, actual: Tensor2d, predicted: Tensor2d) -> dict[str, float]:
        loss: float = float(self._loss(actual, predicted))

        self._accumulator.add(loss)
        self._count += 1
        
        return {
            "value": loss
        }
        
    def result(self) -> dict[str, float]:
        return {
            "value": self._accumulator.value() / self._count
        }
