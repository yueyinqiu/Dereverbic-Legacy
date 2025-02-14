from .i0 import *
from .static_class import StaticClass

class RirAcousticFeatureExtractor(StaticClass):
    _TensorNd = TypeVar("_TensorNd",   # pylint: disable=un-declared-variable
                        Tensor, Tensor1d, Tensor2d, Tensor3d)
    
    # With reference to https://zhuanlan.zhihu.com/p/430228694
    @classmethod
    def energy_decay_curve_decibel(cls,
                                   rir: _TensorNd) -> _TensorNd:
        energy: Tensor = rir ** 2
        energy = energy.flip(-1).cumsum(-1).flip(-1)
        energy = 10 * energy.log10()
        energy = energy - energy[..., 0:1]
        return energy

    # With reference to https://zhuanlan.zhihu.com/p/430228694
    @classmethod
    def get_reverberation_time(cls,
                               energy_decay_curve_decibel: Tensor,
                               decay_decibel: float = 30.,
                               sample_rate: int = 1, 
                               headroom_decibel: float = -5.):
        search: Tensor = torch.tensor([headroom_decibel, headroom_decibel - decay_decibel],
                                      dtype=energy_decay_curve_decibel.dtype,
                                      device=energy_decay_curve_decibel.device)
        search_shape: list[int] = list(energy_decay_curve_decibel.shape)
        search_shape[-1] = 2
        search = search.expand(search_shape)
        
        found: Tensor = torch.searchsorted(energy_decay_curve_decibel.flip(-1), search)
        difference: Tensor = found[..., 0] - found[..., 1]
        return difference / sample_rate

    @classmethod
    def get_reverberation_time_1d(cls,
                                  energy_decay_curve_decibel: Tensor1d,
                                  decay_decibel: float = 30.,
                                  sample_rate: int = 1, 
                                  headroom_decibel: float = -5.) -> Tensor0d:
        return Tensor0d(cls.get_reverberation_time(energy_decay_curve_decibel, 
                                                   decay_decibel, 
                                                   sample_rate, 
                                                   headroom_decibel))

    @classmethod
    def get_reverberation_time_2d(cls,
                                  energy_decay_curve_decibel: Tensor2d,
                                  decay_decibel: float = 30.,
                                  sample_rate: int = 1, 
                                  headroom_decibel: float = -5.) -> Tensor1d:
        return Tensor1d(cls.get_reverberation_time(energy_decay_curve_decibel, 
                                                   decay_decibel, 
                                                   sample_rate, 
                                                   headroom_decibel))


def _test():
    t: Tensor2d = Tensor2d(torch.arange(15000).expand([32, 15000]) / 16000)
    rir: Tensor2d = Tensor2d(torch.exp(-t / 0.1) * torch.sin(10000 * t))
    rir = Tensor2d(torch.cat([torch.zeros([32, 500]), rir, torch.zeros([32, 500])], dim=1))
    matplotlib.pyplot.subplot(121)
    matplotlib.pyplot.plot(rir[0, :])

    edc: Tensor2d = RirAcousticFeatureExtractor.energy_decay_curve_decibel(rir)
    matplotlib.pyplot.subplot(122)
    matplotlib.pyplot.plot(edc[0, :])

    t60: Tensor1d = RirAcousticFeatureExtractor.get_reverberation_time_2d(edc, 60, 16000)
    print(t60)
    # for such ideal rir, expected result: 60 / 20 * 0.1 * ln(10) = 0.6908
    # but due to the lack of "t > (15000 / 16000)" part, it will not be the exact value

    matplotlib.pyplot.savefig("RirAcousticFeatureExtractor.gitignored.png")


if __name__ == "__main__":
    _test()
