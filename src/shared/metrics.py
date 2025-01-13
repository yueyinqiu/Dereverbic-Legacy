from .imports import *
from .dimension_descriptors import *


class STFTLoss(torch.nn.Module):
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

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
    ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length

        self.window: Tensor
        self.register_buffer("window", getattr(torch, window)(win_length), False)

    def forward(self, x: Tensor2d[DBatch, DSample], y: Tensor2d[DBatch, DSample]):
        x_mag: Tensor = STFTLoss.stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag: Tensor = STFTLoss.stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss: Tensor = STFTLoss.spectral_convergence_loss(x_mag, y_mag)
        log_mag_loss: Tensor = STFTLoss.log_stft_magnitude_loss(x_mag, y_mag)
        return sc_loss, log_mag_loss


class MultiResolutionStftLoss(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[64, 512, 2048, 8192],
        hop_sizes=[32, 256, 1024, 4096],
        win_lengths=[64, 512, 2048, 8192],
        window="hann_window",
        sc_weight=1.0,
        mag_weight=1.0,
    ):
        super(MultiResolutionStftLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        fs: int
        ss: int
        wl: int
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses = self.stft_losses + [STFTLoss(fs, ss, wl, window)]
        
        self.zero: Tensor0d
        self.register_buffer("zero", torch.zeros([]), False)

    class Return(TypedDict):
        total: Tensor0d
        sc_loss: Tensor0d
        mag_loss: Tensor0d

    def forward(self, x: Tensor2d[DBatch, DSample], y: Tensor2d[DBatch, DSample]) -> Return:
        """
        shape: [batch_size, rir_length]
        """
        sc_loss: Tensor0d = self.zero
        mag_loss: Tensor0d = self.zero

        f: torch.nn.Module
        for f in self.stft_losses:
            sc_l: Tensor0d
            mag_l: Tensor0d
            sc_l, mag_l = f(x, y)
            sc_loss = Tensor0d(sc_loss + sc_l)
            mag_loss = Tensor0d(mag_loss + mag_l)

        return {
            "total": Tensor0d((sc_loss * self.sc_weight + mag_loss * self.mag_weight) / len(self.stft_losses)),
            "sc_loss": Tensor0d(sc_loss / len(self.stft_losses)),
            "mag_loss": Tensor0d(mag_loss / len(self.stft_losses)),
        }
    