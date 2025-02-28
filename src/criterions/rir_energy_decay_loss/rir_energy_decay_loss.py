from typing import NamedTuple
from statictorch import Tensor0d, Tensor2d
import torch

from audio_processors.rir_acoustic_features import RirAcousticFeatures
from criterions.stft_losses.mrstft_loss_module import MrstftLossModule
from criterions.stft_losses.stft_window import StftWindow


class RirEnergyDecayLoss():
    @staticmethod
    def energy_decay(rir: Tensor2d) -> Tensor2d:
        energy: torch.Tensor = rir ** 2
        energy = energy.flip(-1)
        energy = energy.cumsum(-1)
        energy = energy.clamp_min(1e-6)
        return Tensor2d(10 * energy.log10())

    def __call__(self, actual: Tensor2d, predicted: Tensor2d):
        actual_energy: torch.Tensor = RirEnergyDecayLoss.energy_decay(actual)
        predicted_energy: torch.Tensor = RirEnergyDecayLoss.energy_decay(predicted)
        return torch.nn.functional.l1_loss(predicted_energy, actual_energy)
