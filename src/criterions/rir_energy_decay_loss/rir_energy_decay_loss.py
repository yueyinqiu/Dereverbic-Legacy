from statictorch import Tensor0d, Tensor2d
import torch


class RirEnergyDecayLoss():
    @staticmethod
    def energy_decay(rir: Tensor2d) -> Tensor2d:
        energy: torch.Tensor = rir ** 2
        energy = energy.flip(-1)
        energy = energy.cumsum(-1)
        energy = energy.clamp_min(1e-8)
        return Tensor2d(energy.log10())

    def __call__(self, actual: Tensor2d, predicted: Tensor2d) -> Tensor0d:
        actual_energy: torch.Tensor = RirEnergyDecayLoss.energy_decay(actual)
        predicted_energy: torch.Tensor = RirEnergyDecayLoss.energy_decay(predicted)
        return Tensor0d(torch.nn.functional.l1_loss(predicted_energy, actual_energy))
