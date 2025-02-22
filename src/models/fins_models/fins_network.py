# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

import numpy
import scipy.signal
from statictorch import Tensor2d, Tensor3d
import torch
from torch import Tensor

from models.fins_models.fins_decoder import FinsDecoder
from models.fins_models.fins_encoder import FinsEncoder


class FinsNetwork(torch.nn.Module):
    @staticmethod
    def _get_octave_filters():
        f_bounds: list[tuple[float, float]] = []
        f_bounds.append((22.3, 44.5))
        f_bounds.append((44.5, 88.4))
        f_bounds.append((88.4, 176.8))
        f_bounds.append((176.8, 353.6))
        f_bounds.append((353.6, 707.1))
        f_bounds.append((707.1, 1414.2))
        f_bounds.append((1414.2, 2828.4))
        f_bounds.append((2828.4, 5656.8))
        f_bounds.append((5656.8, 11313.6))
        f_bounds.append((11313.6, 22627.2))

        firs: list = []
        low: float
        high: float
        for low, high in f_bounds:
            fir: numpy.ndarray = scipy.signal.firwin(
                1023,
                numpy.array([low, high]),
                pass_zero="bandpass",  # pyright: ignore [reportArgumentType]
                window="hamming",
                fs=48000,
            )
            firs.append(fir)

        firs_np: numpy.ndarray = numpy.array(firs)
        firs_np = numpy.expand_dims(firs_np, 1)
        return firs_np

    def __init__(self):
        super().__init__()

        rir_length: int = 16000
        early_length: int = 800
        decoder_input_length: int = 134
        num_filters: int = 10
        noise_condition_length: int = 16
        z_size: int = 128
        filter_order: int = 1023

        self.rir_length = rir_length
        self.noise_condition_length = noise_condition_length
        self.num_filters = num_filters

        # Learned decoder input
        self.decoder_input = torch.nn.Parameter(torch.randn((1, 1, decoder_input_length)))
        self.encoder = FinsEncoder()

        self.decoder = FinsDecoder(num_filters, noise_condition_length + z_size, rir_length)

        # Learned "octave-band" like filter
        self.filter = torch.nn.Conv1d(
            num_filters,
            num_filters,
            kernel_size=filter_order,
            stride=1,
            padding="same",
            groups=num_filters,
            bias=False,
        )
        # Octave band pass initialization
        self.filter.weight.data = torch.tensor(FinsNetwork._get_octave_filters(), dtype=torch.float32)

        # Mask for direct and early part
        mask: Tensor3d = Tensor3d(torch.zeros((1, 1, rir_length)))
        mask[:, :, : early_length] = 1.0
        self.mask: Tensor3d
        self.register_buffer("mask", mask, False)

        self.output_conv = torch.nn.Conv1d(num_filters + 1, 1, kernel_size=1, stride=1)

    def forward(self, 
                reverb: Tensor3d, 
                stochastic_noise: Tensor3d, 
                noise_condition: Tensor2d) \
                    -> Tensor3d:
        """
        reverb: [32, 1, 80000]

        stochastic_noise: [32, 10, 16000]

        noise_condition: [32, 16]
        """
        # [32, 10, 16000]
        filtered_noise: Tensor = self.filter(stochastic_noise)
        # [32, 128]
        z: Tensor = self.encoder(reverb)
        # [32, 144 = 128 + 16]
        condition: Tensor = torch.cat([z, noise_condition], dim=-1)
        # [32, 1, 134]
        decoder_input: Tensor = self.decoder_input.repeat(reverb.size()[0], 1, 1)

        # [32, 1, 16000]
        direct_early: Tensor
        # [32, 10, 16000]
        late_mask: Tensor
        direct_early, late_mask = self.decoder(decoder_input, condition)
        # [32, 10, 16000]
        late_part: Tensor = filtered_noise * late_mask
        # [32, 1, 16000]
        direct_early = torch.mul(direct_early, self.mask)
        # [32, 11, 16000]
        rir: Tensor = torch.cat((direct_early, late_part), 1)

        # [32, 1, 16000]
        rir = self.output_conv(rir)
        return Tensor3d(rir)
