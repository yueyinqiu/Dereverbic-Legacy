from typing import NamedTuple
from statictorch import Tensor0d, Tensor2d
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from criterions.stft_losses.mrstft_loss import MrstftLoss
from metrics.metric import Metric


class MrstftLossMetric(Metric):
    def __init__(self, criterion: MrstftLoss):
        self._criterion = criterion
        self._mag_loss = KahanAccumulator()
        self._sc_loss = KahanAccumulator()
    
    @staticmethod
    def for_speech(device: torch.device):
        return MrstftLossMetric(MrstftLoss.for_speech(device))
    
    @staticmethod
    def for_rir(device: torch.device):
        return MrstftLossMetric(MrstftLoss.for_rir(device))

    def append(self, actual: Tensor2d, predicted: Tensor2d) -> dict[str, float]:
        losses: MrstftLoss.Return = self._criterion(actual, predicted)
        mag_loss: float = float(losses.mag_loss)
        sc_loss: float = float(losses.sc_loss)

        self._mag_loss.add(mag_loss)
        self._sc_loss.add(sc_loss)

        return {
            "total": mag_loss + sc_loss,
            "mag": mag_loss,
            "sc": sc_loss
        }
        
    def result(self) -> dict[str, float]:
        return {
            "total": self._mag_loss.value() + self._sc_loss.value(),
            "mag": self._mag_loss.value(),
            "sc": self._sc_loss.value()
        }
