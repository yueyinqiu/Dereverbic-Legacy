from typing import Any, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]

from criterions.rir_energy_decay_loss.rir_energy_decay_loss import RirEnergyDecayLoss
from criterions.stft_losses.mrstft_loss import MrstftLoss
from models.ricbe_models.networks.tdunet_dbe_network import TdunetDbeNetwork
from trainers.trainable import Trainable


class TdunetDbeModel(Trainable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = TdunetDbeNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)

        self.mrstft = MrstftLoss.for_rir(device)
        self.l1 = torch.nn.L1Loss()
        self.energy_decay = RirEnergyDecayLoss()

        self.train_preparation: Tensor0d | None = None

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _predict(self,
                 reverb_batch: Tensor2d) -> Tensor2d:
        rir: Tensor3d = self.module(reverb_batch.unsqueeze(1))
        return Tensor2d(rir.squeeze(1))

    def _calculate_losses(self, 
                          actual: Tensor2d,
                          predicted: Tensor2d) -> tuple[Tensor0d, dict[str, float]]:
        mrstft: MrstftLoss.Return = self.mrstft(actual, predicted)
        l1: torch.Tensor = self.l1(actual, predicted)
        energy: torch.Tensor = self.energy_decay(actual, predicted)

        total: Tensor0d = Tensor0d(mrstft.total() + l1 + energy)
        return total, {
            "loss_total": float(total),
            "loss_mrstft_mag": float(mrstft.mag_loss),
            "loss_mrstft_sc": float(mrstft.sc_loss),
            "loss_l1": float(l1),
            "loss_energy_decay": float(energy)
        }

    def prepare_train_on(self, 
                         reverb_batch: Tensor2d, 
                         rir_batch: Tensor2d, 
                         speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch)
        
        losses: dict[str, float]
        self.train_preparation, losses = self._calculate_losses(rir_batch, predicted)

        return losses

    def train_prepared(self):
        assert self.train_preparation is not None
        self.optimizer.zero_grad()
        self.train_preparation.backward()
        self.optimizer.step()
    
    def evaluate_on(self, 
                    reverb_batch: Tensor2d):
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
    def validate_on(self,
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        self.module.eval()

        predicted: Tensor2d = self._predict(reverb_batch)
        losses: dict[str, float]
        _, losses = self._calculate_losses(rir_batch, predicted)

        self.module.train()
        
        return losses["loss_total"], losses
