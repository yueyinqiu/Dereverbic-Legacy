from typing import Generic, TypeVar
import typing
import matplotlib.pyplot
from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d, anify
from torch import Tensor
import torch

_T = TypeVar("_T", Tensor, Tensor1d, Tensor2d, Tensor3d)   # pylint: disable=un-declared-variable
_TMinusOne = TypeVar("_TMinusOne", Tensor, Tensor0d, Tensor1d, Tensor2d)   # pylint: disable=un-declared-variable

class _RirAcousticFeatures(Generic[_T, _TMinusOne]):
    def __init__(self, rir: _T) -> None:
        self._rir: _T = rir
        self._instantaneous_energy: _T | None = None
        self._energy_decay_curve: _T | None = None
        self._energy_decay_curve_decibel: _T | None = None
        self._direct_sound_indices: _TMinusOne | None = None
    
    def rir(self) -> _T:
        return self._rir
    
    def instantaneous_energy(self) -> _T:
        if self._instantaneous_energy is None:
            self._instantaneous_energy = typing.cast(_T, self.rir() ** 2)
        return self._instantaneous_energy
    
    def energy_decay_curve(self):
        if self._energy_decay_curve is None:
            self._energy_decay_curve = typing.cast(_T, 
                                                   self.instantaneous_energy().flip(-1).cumsum(-1).flip(-1))
        return self._energy_decay_curve
    
    # With reference to https://zhuanlan.zhihu.com/p/430228694
    def energy_decay_curve_decibel(self):
        if self._energy_decay_curve_decibel is None:
            energy: Tensor = 10 * self.energy_decay_curve().log10()
            self._energy_decay_curve_decibel = typing.cast(_T, energy - energy[..., 0:1])
        return self._energy_decay_curve_decibel
    
    # With reference to https://zhuanlan.zhihu.com/p/430228694
    def reverberation_time(self,
                           decay_decibel: float = 30.,
                           sample_rate: int = 1, 
                           headroom_decibel: float = -5.) -> _TMinusOne:
        energy_decay_curve_decibel: Tensor = self.energy_decay_curve_decibel()

        search: Tensor = torch.tensor([headroom_decibel, headroom_decibel - decay_decibel],
                                      dtype=energy_decay_curve_decibel.dtype,
                                      device=energy_decay_curve_decibel.device)
        search_shape: list[int] = list(energy_decay_curve_decibel.shape)
        search_shape[-1] = 2
        search = search.expand(search_shape)
        
        found: Tensor = torch.searchsorted(energy_decay_curve_decibel.flip(-1), search)
        difference: Tensor = found[..., 0] - found[..., 1]
        return anify(difference / sample_rate)
    
    def direct_sound_indices(self) -> _TMinusOne:
        if self._direct_sound_indices is None:
            self._direct_sound_indices = typing.cast(_TMinusOne, 
                                                     self.instantaneous_energy().argmax(dim=-1))
        return self._direct_sound_indices

    # With reference to https://zhuanlan.zhihu.com/p/430228694
    def direct_to_reverberant_energy_ratio_db(self, split_point: _TMinusOne | None = None) -> _TMinusOne:
        if split_point is None:
            indices: Tensor = self.direct_sound_indices() + 1
        else:
            indices = split_point
        indices = indices.unsqueeze(-1)

        energy_decay_curve: Tensor = self.energy_decay_curve()
        reverberant: Tensor = energy_decay_curve.gather(-1, indices)
        reverberant = reverberant.squeeze(-1)
        total: Tensor = energy_decay_curve[..., 0]
        direct: Tensor = total - reverberant
        return anify(10 * torch.log10(direct / reverberant))


class RirAcousticFeatures1d(_RirAcousticFeatures[Tensor1d, Tensor0d]):
    pass


class RirAcousticFeatures2d(_RirAcousticFeatures[Tensor2d, Tensor1d]):
    pass


def _t60_test():
    t: Tensor2d = Tensor2d(torch.arange(15000).expand([32, 15000]) / 16000)
    rir: Tensor2d = Tensor2d(torch.exp(-t / 0.1) * torch.sin(10000 * t))
    rir = Tensor2d(torch.cat([torch.zeros([32, 500]), rir, torch.zeros([32, 500])], dim=1))
    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.plot(rir[0, :])

    features: RirAcousticFeatures2d = RirAcousticFeatures2d(rir)
    matplotlib.pyplot.subplot(122)
    matplotlib.pyplot.plot(features.energy_decay_curve_decibel()[0, :])

    t60: Tensor1d = features.reverberation_time(60, 16000)
    print(t60)
    # for such ideal rir, expected result: 60 / 20 * 0.1 * ln(10) = 0.6908
    # but due to the lack of "t > (15000 / 16000)" part, it will not be the exact value

    matplotlib.pyplot.savefig("RirAcousticFeatureExtractor.gitignored.png")


def _drr_test():
    rir: Tensor = torch.tensor([
        [
            [0, 8, 10, 7, 5, 3, 2, 1, 0, 0],
            # energy: 0 64 100 49 25 9 4 1 0 0
            # direct sound index: 2 (10)
            # split point: 3
            # drr: (64 + 100) / (49 + 25 + 9 + 4 + 1) = 1.8636363
            # drr_db: 10 lg(1.8636363) = 2.704
            [0, 6, 8, 10, 2, 1, 0, 0, 0, 0]
            # energy: 0 36 64 100 4 1 0 0 0 0
            # direct sound index: 3 (10)
            # split point: 4
            # drr: (36 + 64 + 100) / (4 + 1) = 40
            # drr_db: 10 log(40) = 16.021
        ],
        [
            [0, 8, 10, 7, 5, 3, 2, 1, 0, 0],
            [0, 6, 8, 10, 2, 1, 0, 0, 0, 0]
        ]
    ], dtype=torch.float)
    features: _RirAcousticFeatures = _RirAcousticFeatures(rir)
    print(features.direct_sound_indices())
    print(features.direct_to_reverberant_energy_ratio_db())


if __name__ == "__main__":
    _t60_test()
    _drr_test()
