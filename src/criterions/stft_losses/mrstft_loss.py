from typing import NamedTuple
from statictorch import Tensor0d, Tensor2d
import torch

from criterions.stft_losses.mrstft_loss_module import MrstftLossModule
from criterions.stft_losses.stft_window import StftWindow


class MrstftLoss():
    def __init__(self, 
                 device: torch.device,
                 fft_sizes: list[int],
                 hop_sizes: list[int],
                 win_lengths: list[int],
                 window: StftWindow) -> None:
        self._module = MrstftLossModule(fft_sizes, hop_sizes, win_lengths, window).to(device)
    
    @staticmethod
    def for_speech(device: torch.device):
        return MrstftLoss(device, 
                          fft_sizes=[512, 1024, 2048],
                          hop_sizes=[50, 120, 240],
                          win_lengths=[240, 600, 1200], 
                          window="hann_window")
    
    @staticmethod
    def for_rir(device: torch.device):
        return MrstftLoss(device, 
                          fft_sizes=[32, 256, 1024, 4096],
                          hop_sizes=[16, 128, 512, 2048],
                          win_lengths=[32, 256, 1024, 4096], 
                          window="hann_window")

    class Return(NamedTuple):
        sc_loss: Tensor0d
        mag_loss: Tensor0d

        def total(self, factor_sc: float = 1., factor_mag: float = 1.):
            return Tensor0d(factor_sc * self.sc_loss + factor_mag * self.mag_loss)

    def __call__(self, x: Tensor2d, y: Tensor2d) -> Return:
        sc_loss: Tensor0d
        mag_loss: Tensor0d
        sc_loss, mag_loss = self._module(x, y)
        return MrstftLoss.Return(sc_loss, mag_loss)