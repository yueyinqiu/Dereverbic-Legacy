from statictorch import Tensor1d
from torch import Tensor
import torch

from metrics.metric import Metric


class PearsonCorrelationMetric(Metric[Tensor1d]):
    def __init__(self):
        self._actual: Tensor1d = Tensor1d(torch.empty([0], device="cpu"))
        self._predicted: Tensor1d = Tensor1d(torch.empty([0], device="cpu"))
    
    def _calculate_on(self, actual: Tensor1d, predicted: Tensor1d):
        stack: Tensor = torch.stack([actual, predicted], dim=0)
        corrcoef: Tensor = torch.corrcoef(stack)
        return corrcoef[0, 1]

    def append(self, actual: Tensor1d, predicted: Tensor1d) -> dict[str, float]:
        self._actual = Tensor1d(
            torch.cat([self._actual, actual.detach().cpu()], dim=0))
        self._predicted = Tensor1d(
            torch.cat([self._predicted, predicted.detach().cpu()], dim=0))
        return {
            "value": float(self._calculate_on(actual, predicted))
        }
        
    def result(self) -> dict[str, float]:
        return {
            "value": float(self._calculate_on(self._actual, self._predicted))
        }
