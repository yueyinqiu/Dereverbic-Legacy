from statictorch import Tensor2d
from torch import Tensor
import torch

from criterions.stft_losses.stft_window import StftWindow


class StftLoss(torch.nn.Module):
    @staticmethod
    def stft(x: Tensor, fft_size: int, hop_size: int, win_length: int, window: Tensor):
        x_stft: Tensor = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        x_mag: Tensor = torch.sqrt(torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8))
        return x_mag

    @staticmethod
    def spectral_convergence_loss(x_mag: Tensor, y_mag: Tensor):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    @staticmethod
    def log_stft_magnitude_loss(x_mag: Tensor, y_mag: Tensor):
        return torch.nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def __init__(self,
                 fft_size: int, 
                 shift_size: int, 
                 win_length: int,
                 window: StftWindow):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        
        self.window: Tensor
        self.register_buffer("window", getattr(torch, window)(win_length), False)

    def forward(self, x: Tensor2d, y: Tensor2d):
        x_mag: Tensor = StftLoss.stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag: Tensor = StftLoss.stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss: Tensor = StftLoss.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss: Tensor = StftLoss.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, log_mag_loss

