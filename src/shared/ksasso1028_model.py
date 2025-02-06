# https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

import torch
import torch.nn as nn

class AutoVerb(nn.Module):
    def __init__(self, blocks: int, in_channels: int, channel_factor: int):
        super(AutoVerb, self).__init__()
        start: int = in_channels
        self.conv = nn.Conv1d(1, in_channels, kernel_size=5, padding=2, stride=1)
        self.prelu = nn.PReLU(in_channels)

        # encoder
        self.hidden= nn.ModuleList()

        # decoder
        self.decode = nn.ModuleList()

        dil: int = 1
        for _ in range(blocks):
            self.hidden.append(Encoder(start, stride=4, dil=dil, mul=channel_factor))
            # set channel size for next block
            start = start + channel_factor

        self.bottleneck = nn.Conv1d(start, start, 3, padding=(((3 - 1) // 2) * dil), dilation=dil)
        self.bottle_act = nn.PReLU(start)
        self.lstm=  nn.LSTM(start, start, num_layers=2, batch_first = True)
        self.linear = nn.Linear(start, start)

        for _ in range(blocks):
            self.decode.append(Upscale(start, stride=1, dil=dil, mul=channel_factor))
            start = start - channel_factor
        self.process = CutBlock(start, 1)

    def forward(self, mix):
        # [32, 1, 80000]
        x = mix.clone()
        # [32, 48, 80000]
        x = self.prelu(self.conv(x))

        # [32, 48, 80000]
        # [32, 96, 20000]
        # [32, 144, 5000]
        # [32, 192, 1250]
        # [32, 240, 313]
        # [32, 288, 79]
        features = []
        features.append(x)
        for module in self.hidden:
            x = module(x)
            features.append(x)

        # [32, 288, 79]
        bottle_neck = x.clone()
        # [32, 288, 79]
        bottle_neck = self.bottle_act(self.bottleneck(bottle_neck))
        # [32, 79, 288]
        bottle_neck = bottle_neck.permute(0, 2, 1)
        # [32, 79, 288]
        bottle_neck, _ = self.lstm(bottle_neck)
        # [32, 79, 288]
        bottle_neck = self.linear(bottle_neck)
        # [32, 288, 79]
        bottle_neck = bottle_neck.permute(0, 2, 1)

        # [32, 288, 79]
        x = x + bottle_neck
        for i, module in enumerate(self.decode):
            index = i + 1
            # Match dims from encoder to decoder
            x = x[:, :, :features[-abs(index)].size(2)] + features[-abs(index)]
            x = module(x)  # [32, 240, 316(313)], [32, 240, 1252(1250)], ...
        x = x[:, :, :mix.size(-1)]

        # remove noise, refine synthesis of final waveform.
        out = self.process(x)
        return out


class Encoder(nn.Module):
    def __init__(self, channels, dil=1, stride=1, mul=1, gru=False):
        super(Encoder, self).__init__()
        self.kernel = 7
        self.stride = stride

        self.conv1 = nn.Conv1d(channels,channels + mul, kernel_size = self.kernel, stride = stride,
                               padding = ((self.kernel - 1) // 2) * dil, dilation = dil)
        
        if gru:
            self.gru = nn.GRU(channels + mul, channels + mul, num_layers=2, batch_first = True)
        else:
            self.gru = None

        self.prelu1 = nn.PReLU(channels + mul)

    def forward(self, x):
        x =  self.prelu1(self.conv1(x))
        if self.gru is not None:
            x = x.permute(0,2,1)
            x, _ = self.gru(x)
            x = x.permute(0,2,1)

        return x


class Upscale(nn.Module):
    def __init__(self, channels, dil, stride=1, mul=1):
        super(Upscale, self).__init__()
        self.stride = stride
        self.kernel = 7
        self.upsample = nn.Upsample(scale_factor = 4)
        self.conv1 = nn.Conv1d(channels, channels - mul, kernel_size = self.kernel, dilation = dil,
                               padding=((self.kernel - 1) // 2) * dil)
        self.conv2 = nn.Conv1d(channels - mul, channels - mul, kernel_size = self.kernel, dilation = dil,
                               padding=((self.kernel - 1) // 2) * dil)
        self.conv3 = nn.Conv1d(channels - mul, channels - mul, kernel_size=self.kernel, dilation=dil,
                               padding=((self.kernel - 1) // 2) * dil)
        # padding=dil)

        self.prelu1 = nn.PReLU(channels - mul)
        self.prelu2 = nn.PReLU(channels - mul)
        self.prelu3 = nn.PReLU(channels - mul)


    def forward(self, x):
        x = self.upsample(x)
        x = self.prelu1(self.conv1(x))
        upscaled = x.clone()
        x = self.prelu2((self.conv2(x)))
        x = self.prelu3(self.conv3(x))
        # residual mapping
        x = upscaled + x

        return x


# block of conv layers to refine synthesis
class CutBlock(nn.Module):
    def __init__(self, channels, out, dil=1, stride=1, mul=1,):
        super(CutBlock, self).__init__()
        self.first_kernel = 11
        self.second_kernel = 3
        self.conv1 = nn.Conv1d(channels, channels // 2, kernel_size = self.first_kernel, stride = stride, padding = ((self.first_kernel - 1) // 2) * dil)
        self.conv2 = nn.Conv1d(channels // 2, out, kernel_size = self.second_kernel, stride = stride, padding = ((self.second_kernel - 1) // 2) * dil)
        self.prelu = nn.PReLU(channels // 2)
    def forward(self, x):
        out = (self.prelu(self.conv1(x)))
        out = (self.conv2(out))

        return out

# Experimental sin activation inspired by SIREN paper, not used for final network

class SinScale(nn.Module):
    def __init__(self, in_features):
        super(SinScale, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1, in_features))
        nn.init.uniform_(self.scale, a = 0.1, b = 1.0)  # Initialize scale parameter

    def forward(self, x):
        # learn scaling for each channel
        scaled_x = x.transpose(1,2) * self.scale
        transformed_x = torch.sin(scaled_x)
        transformed_x = transformed_x.transpose(1,2)
        return transformed_x

from .i0 import *
from .rir_blind_estimation_model import RirBlindEstimationModel
# loss function grabbed from CleanUnet - > https://github.com/NVIDIA/CleanUNet/blob/main/stft_loss.py
"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window, return_complex=False
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
            self, fft_size=1024, shift_size=120, win_length=600, window="hann_window",
            band="full"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.band = band

        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        if self.band == "high":
            freq_mask_ind = x_mag.shape[1] // 2  # only select high frequency bands
            sc_loss = self.spectral_convergence_loss(x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :])
            mag_loss = self.log_stft_magnitude_loss(x_mag[:, freq_mask_ind:, :], y_mag[:, freq_mask_ind:, :])
        elif self.band == "full":
            sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
            mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        else:
            raise NotImplementedError

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
            self, fft_sizes=[512, 1024, 2048, 4096], hop_sizes=[50, 120, 240, 480], win_lengths=[512, 1024, 2048, 4096],
            window="hann_window", sc_lambda=.9, mag_lambda=.89, band="full"
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            *_lambda (float): a balancing factor across different losses.
            band (str): high-band or full-band loss
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        self.sc_lambda = sc_lambda
        self.mag_lambda = mag_lambda

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, band)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss = sc_loss + sc_l
            mag_loss = mag_loss + mag_l

        sc_loss = sc_loss * self.sc_lambda
        sc_loss = sc_loss / len(self.stft_losses)
        mag_loss = mag_loss * self.mag_lambda
        mag_loss = mag_loss / len(self.stft_losses)

        return sc_loss, mag_loss
    
class KsassoModel(RirBlindEstimationModel):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = AutoVerb(blocks=5, in_channels=48, channel_factor=48).to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)
        self.random = torch.Generator(device).manual_seed(seed)
        self.spec_loss = MultiResolutionSTFTLoss().to(device)
        self.l1 = nn.L1Loss().to(device)

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        random: Tensor

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "random": self.random.get_state()
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.random.set_state(state["random"])

    def _predict(self,
                 reverb_batch: Tensor2d):
        speech: Tensor3d = self.module(reverb_batch.unsqueeze(1))
        return Tensor2d(speech.squeeze(1)[:16000])

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch)
        loss_l1: Tensor0d = self.l1(predicted, rir_batch)
        loss_stft: Tensor0d
        _, loss_stft = self.spec_loss(predicted, rir_batch)
        loss_total: Tensor0d = Tensor0d(loss_l1 + loss_stft)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(loss_total),
            "loss_l1": float(loss_l1),
            "loss_stft": float(loss_stft)
        }

        return result

    def evaluate_on(self, reverb_batch: Tensor2d):
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch)
        self.module.train()
        return predicted
    