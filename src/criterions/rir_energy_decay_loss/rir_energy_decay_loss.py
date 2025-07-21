from pathlib import Path
from typing import TypeVar
import csdir
import matplotlib
import matplotlib.pyplot
from statictorch import Tensor0d, Tensor1d, Tensor2d, Tensor3d, anify
import torch

# RIR Energy Decay Loss
class RirEnergyDecayLoss():
    _TensorNd = TypeVar("_TensorNd",   # pylint: disable=un-declared-variable
                        torch.Tensor, Tensor1d, Tensor2d, Tensor3d)
    def __init__(self, decibel: bool = True):
        self._decibel = decibel
        
    def _energy_decay(self, rir: _TensorNd) -> _TensorNd:
        energy: torch.Tensor = rir ** 2
        energy = energy.flip(-1)
        energy = energy.cumsum(-1)
        if self._decibel:
            energy = energy.clamp_min(1e-8)
            energy = energy.log10()
        return anify(energy)

    def __call__(self, actual: Tensor2d, predicted: Tensor2d) -> Tensor0d:
        actual_energy: torch.Tensor = self._energy_decay(actual)
        predicted_energy: torch.Tensor = self._energy_decay(predicted)
        return Tensor0d(torch.nn.functional.l1_loss(predicted_energy, actual_energy))
    

def _test():
    outputs: Path = csdir.create_directory(".gitignored/RirEnergyDecayLoss")

    t: torch.Tensor = torch.arange(15000) / 16000
    rir: torch.Tensor = torch.exp(-t / 0.1) * torch.sin(10000 * t)
    rir = torch.cat([torch.zeros([500]), rir, torch.zeros([500])], dim=0)
    matplotlib.pyplot.xlim(0, 16000)
    matplotlib.pyplot.plot(rir)
    matplotlib.pyplot.savefig(outputs / "rir.png")
    matplotlib.pyplot.clf()

    ed: torch.Tensor = RirEnergyDecayLoss()._energy_decay(rir)
    matplotlib.pyplot.xlim(16000, 0)
    matplotlib.pyplot.plot(ed)
    matplotlib.pyplot.savefig(outputs / "ed.png")
    matplotlib.pyplot.clf()


if __name__ == "__main__":
    _test()
