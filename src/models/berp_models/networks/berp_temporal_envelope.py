import torch
import torchaudio

from models.berp_models.networks.berp_butter_worth_filter import BerpButterWorthFilter
from models.berp_models.networks.berp_hilbert_transform import BerpHilbertTransform


class BerpTemporalEnvelope(torch.nn.Module):
    def __init__(
        self, dim: int, fs: int = 16000, mode: str = "envelope", fc: int = 128
    ):
        super().__init__()
        self.fc = fc
        self.fs = fs
        self.mode = mode

        self.hilbert = BerpHilbertTransform(axis=dim)
        self.Bd, self.Ad = BerpButterWorthFilter(fs=self.fs, fc=self.fc).lpf()

    def hilbert_filt(self, x: torch.Tensor):
        analytic_signal: torch.Tensor = self.hilbert(x)
        amplitude_envelope: torch.Tensor = torch.abs(analytic_signal)
        return amplitude_envelope

    def TAE(self, x):
        x = self.hilbert_filt(x)
        x = torchaudio.functional.filtfilt(x, self.Ad.to(x.device), self.Bd.to(x.device))
        return x

    def envelope(self, x):
        return self.hilbert_filt(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "envelope":
            return self.envelope(x).float()
        elif self.mode == "TAE":
            return self.TAE(x).float()
        else:
            raise ValueError("mode must be 'envelope' or 'TAE'.")