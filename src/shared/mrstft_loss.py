# The metric is modified from: 
# https://github.com/kyungyunlee/fins/blob/main/fins/loss.py
# Please respect the original license

from .i0 import *


_Window: TypeAlias = Literal[
    "hann_window", 
    "kaiser_window", 
    "hamming_window", 
    "bartlett_window", 
    "blackman_window"
]


class _StftLoss(torch.nn.Module):
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
                 window: _Window):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        
        self.window: Tensor
        self.register_buffer("window", getattr(torch, window)(win_length), False)

    def forward(self, x: Tensor2d, y: Tensor2d):
        x_mag: Tensor = _StftLoss.stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag: Tensor = _StftLoss.stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss: Tensor = _StftLoss.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss: Tensor = _StftLoss.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, log_mag_loss


class _MrstftLossModule(torch.nn.Module):
    def __init__(self, 
                 fft_sizes: list[int],
                 hop_sizes: list[int],
                 win_lengths: list[int],
                 window: _Window):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes

        fs: int
        ss: int
        wl: int
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses = self.stft_losses + [_StftLoss(fs, ss, wl, window)]
        
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
    

class MrstftLoss():
    def __init__(self, 
                 device: torch.device,
                 fft_sizes: list[int],
                 hop_sizes: list[int],
                 win_lengths: list[int],
                 window: _Window) -> None:
        self._module = _MrstftLossModule(fft_sizes, hop_sizes, win_lengths, window).to(device)
    
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