# The criterion is modified from:
# https://github.com/kyungyunlee/fins
# Please respect the original license

from statictorch import Tensor0d, Tensor2d
import torch

from criterions.stft_losses.stft_loss import StftLoss
from criterions.stft_losses.stft_window import StftWindow


class MrstftLossModule(torch.nn.Module):
    def __init__(self, 
                 fft_sizes: list[int],
                 hop_sizes: list[int],
                 win_lengths: list[int],
                 window: StftWindow):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes

        fs: int
        ss: int
        wl: int
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses = self.stft_losses + [StftLoss(fs, ss, wl, window)]
        
        self.zero: Tensor0d
        self.register_buffer("zero", torch.zeros([]), False)

    def forward(self, x: Tensor2d, y: Tensor2d) -> tuple[Tensor0d, Tensor0d]:
        sc_loss: Tensor0d = self.zero
        mag_loss: Tensor0d = self.zero

        f: torch.nn.Module
        for f in self.stft_losses:
            sc_l: Tensor0d
            mag_l: Tensor0d
            sc_l, mag_l = f(x, y)
            sc_loss = Tensor0d(sc_loss + sc_l)
            mag_loss = Tensor0d(mag_loss + mag_l)

        sc_loss = Tensor0d(sc_loss / len(self.stft_losses))
        mag_loss = Tensor0d(mag_loss / len(self.stft_losses))
        return sc_loss, mag_loss
    