# This model is modified from (without modification of its main behavior):
# https://github.com/Alizeded/BERP
# (The only behavior change: The returned RIRs will be of the given length exactly.)
# The original repository is licensed under GPL-3.0
# Please also respect the original author's rights

import numpy
import torch
from torch import Tensor


class BerpSparseStochasticIr:
    def __init__(self, mu_Th: float, seed: int, fs: int = 16000, rir_length: int = 16000):
        super().__init__()
        self.mu_Th = mu_Th
        self.fs = fs
        self.seed = seed
        self.rir_length = rir_length

    def __call__(self, Ti: float, Td: float, volume: float) -> Tensor:
        volume = int(round(volume, 0)) + 1
        early_reflection_range: Tensor = torch.arange(-Ti, -1 / self.fs, 1 / self.fs)
        late_reverberation_range: Tensor = torch.linspace(0, 
                                                          self.rir_length / self.fs - Ti, 
                                                          self.rir_length - len(early_reflection_range))
        early_reflection_part: Tensor = torch.exp(6.9 * (early_reflection_range / Ti))
        early_reflection_carrier: numpy.ndarray = numpy.random.default_rng(self.seed).normal(
            self.mu_Th, 1.0, size=len(early_reflection_range)
        )
        early_reflection_carrier_poisson: numpy.ndarray = numpy.random.default_rng(self.seed).poisson(
            self.mu_Th, size=volume
        )
        if len(early_reflection_range) > volume:
            early_reflection_carrier[:volume] = early_reflection_carrier_poisson
        elif len(early_reflection_range) < volume:
            early_reflection_carrier = early_reflection_carrier_poisson[
                : len(early_reflection_range)
            ]
        elif len(early_reflection_range) == volume:
            early_reflection_carrier = early_reflection_carrier_poisson

        early_reflection: Tensor = early_reflection_part * torch.from_numpy(early_reflection_carrier).float()

        late_reverberation_part: Tensor = torch.exp(-6.9 * (late_reverberation_range / Td))
        late_reverberation_carrier: numpy.ndarray = numpy.random.default_rng(self.seed).normal(
            0.0, 1.0, size=len(late_reverberation_range)
        )
        late_reverberation: Tensor = late_reverberation_part * torch.from_numpy(late_reverberation_carrier).float()
        synthesized_rir: Tensor = torch.cat((early_reflection, late_reverberation), dim=0)

        synthesized_rir_envelope: Tensor = torch.cat(
            (early_reflection_part, late_reverberation_part), dim=0
        )
        b: Tensor = (1 / torch.trapz(synthesized_rir_envelope)).sqrt()
        return b * synthesized_rir
