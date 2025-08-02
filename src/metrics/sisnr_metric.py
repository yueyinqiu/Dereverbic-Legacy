from statictorch import Tensor2d
from torch import Tensor
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from metrics.metric import Metric


class SisnrMetric(Metric[Tensor2d]):
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


def _test():
    actual: Tensor2d = Tensor2d(torch.tensor([[3.0, -0.5, 2.0, 7.0], [3.0, -0.5, 2.0, 7.0]]))
    predicted: Tensor2d = Tensor2d(torch.tensor([[2.5, 0.0, 2.0, 8.0], [2.5, 0.0, 2.0, 8.0]]))

    metric: SisnrMetric = SisnrMetric()
    print(metric.append(actual, predicted))
    print(metric.append(predicted, actual))
    print(metric.result())


if __name__ == "__main__":
    _test()
