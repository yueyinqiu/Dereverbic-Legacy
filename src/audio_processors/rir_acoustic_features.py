from typing import TypeVar
import matplotlib.pyplot
from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d, anify
from torch import Tensor
import torch
from basic_utilities.static_class import StaticClass


class RirAcousticFeatures(StaticClass):
    _TensorNd = TypeVar("_TensorNd",   # pylint: disable=un-declared-variable
                        Tensor, Tensor1d, Tensor2d, Tensor3d)
    
    # With reference to https://zhuanlan.zhihu.com/p/430228694
    @classmethod
    def _energy_decay_curve_decibel(cls,
                                    rir: _TensorNd) -> _TensorNd:
        energy: Tensor = rir ** 2
        energy = energy.flip(-1).cumsum(-1).flip(-1)
        energy = 10 * energy.log10()
        energy = energy - energy[..., 0:1]
        return anify(energy)

    # With reference to https://zhuanlan.zhihu.com/p/430228694
    @classmethod
    def get_reverberation_time(cls,
                               rir: Tensor,
                               decay_decibel: float = 30.,
                               sample_rate: int = 1, 
                               headroom_decibel: float = -5.):
        energy_decay: Tensor = cls._energy_decay_curve_decibel(rir)
        search: Tensor = torch.tensor([headroom_decibel, headroom_decibel - decay_decibel],
                                      dtype=energy_decay.dtype,
                                      device=energy_decay.device)
        search_shape: list[int] = list(energy_decay.shape)
        search_shape[-1] = 2
        search = search.expand(search_shape)
        
        found: Tensor = torch.searchsorted(energy_decay.flip(-1), search)
        difference: Tensor = found[..., 0] - found[..., 1]
        return difference / sample_rate
    
    @classmethod
    def get_reverberation_time_1d(cls,
                                  rir: Tensor1d,
                                  decay_decibel: float = 30.,
                                  sample_rate: int = 1, 
                                  headroom_decibel: float = -5.) -> Tensor0d:
        return Tensor0d(cls.get_reverberation_time(rir, 
                                                   decay_decibel, 
                                                   sample_rate, 
                                                   headroom_decibel))

    @classmethod
    def get_reverberation_time_2d(cls,
                                  rir: Tensor2d,
                                  decay_decibel: float = 30.,
                                  sample_rate: int = 1, 
                                  headroom_decibel: float = -5.) -> Tensor1d:
        return Tensor1d(cls.get_reverberation_time(rir, 
                                                   decay_decibel, 
                                                   sample_rate, 
                                                   headroom_decibel))
    



def _test():
    t: Tensor2d = Tensor2d(torch.arange(15000).expand([32, 15000]) / 16000)
    rir: Tensor2d = Tensor2d(torch.exp(-t / 0.1) * torch.sin(10000 * t))
    rir = Tensor2d(torch.cat([torch.zeros([32, 500]), rir, torch.zeros([32, 500])], dim=1))
    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.plot(rir[0, :])

    edc: Tensor2d = RirAcousticFeatures._energy_decay_curve_decibel(rir)
    matplotlib.pyplot.subplot(122)
    matplotlib.pyplot.plot(edc[0, :])

    t60: Tensor1d = RirAcousticFeatures.get_reverberation_time_2d(edc, 60, 16000)
    print(t60)
    # for such ideal rir, expected result: 60 / 20 * 0.1 * ln(10) = 0.6908
    # but due to the lack of "t > (15000 / 16000)" part, it will not be the exact value

    matplotlib.pyplot.savefig("RirAcousticFeatureExtractor.gitignored.png")


if __name__ == "__main__":
    _test()
