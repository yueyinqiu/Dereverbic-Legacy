from typing import Any, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]

from criterions.stft_losses.mrstft_loss import MrstftLoss
from models.ricbe_models.networks.ricbe_full_network import RicbeFullNetwork
from trainers.trainable import Trainable
from trainers.validatable import Validatable


class RicbeFullModel(Trainable, Validatable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = RicbeFullNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)

        self.mrstft = MrstftLoss.for_rir(device)

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

        total: Tensor0d = mrstft.total()
        return total, {
            "loss_total": float(total),
            "loss_mrstft_mag": float(mrstft.mag_loss),
            "loss_mrstft_sc": float(mrstft.sc_loss)
        }

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch)
        
        losses: dict[str, float]
        loss_total: Tensor0d
        loss_total, losses = self._calculate_losses(rir_batch, predicted)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return losses

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
