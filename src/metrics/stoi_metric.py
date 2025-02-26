import batch_pystoi
import numpy
from statictorch import Tensor2d
from basic_utilities.kahan_accumulator import KahanAccumulator
from metrics.metric import Metric


class StoiMetric(Metric[Tensor2d]):
    def __init__(self, sample_rate: int):
        self._sample_rate = sample_rate

        self._accumulator = KahanAccumulator()
        self._count = 0
    
    def append(self, actual: Tensor2d, predicted: Tensor2d) -> dict[str, float]:
        actual_numpy: numpy.ndarray = actual.detach().cpu().numpy()
        predicted_numpy: numpy.ndarray = predicted.detach().cpu().numpy()

        values: numpy.ndarray = batch_pystoi.stoi(actual_numpy, 
                                                  predicted_numpy, 
                                                  self._sample_rate)
        mean: float = float(numpy.mean(values))
        
        self._accumulator.add(mean)
        self._count += 1

        return {
            "value": mean
        }
        
    def result(self) -> dict[str, float]:
        return {
            "value": self._accumulator.value() / self._count
        }
