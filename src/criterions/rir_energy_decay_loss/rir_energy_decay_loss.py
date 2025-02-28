from typing import NamedTuple
from statictorch import Tensor0d, Tensor2d
import torch

from audio_processors.rir_acoustic_features import RirAcousticFeatures
from criterions.stft_losses.mrstft_loss_module import MrstftLossModule
from criterions.stft_losses.stft_window import StftWindow


class RirEnergyDecayLoss():
    def __call__(self, actual: Tensor2d, predicted: Tensor2d):
        actual_energy: torch.Tensor = (actual ** 2).flip(-1).cumsum(-1).flip(-1).log10()
        predicted_energy: torch.Tensor = (actual ** 2).flip(-1).cumsum(-1).flip(-1).log10()
        return torch.nn.functional.l1_loss(predicted_energy, actual_energy)
