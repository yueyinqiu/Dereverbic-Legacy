# This model is modified from: 
# https://github.com/NVIDIA/CleanUNet
# Please respect the original license

from typing import Any, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from torch.optim import Adam  # pyright: ignore [reportPrivateImportUsage]

from criterions.stft_losses.mrstft_loss import MrstftLoss
from models.cleanunet_models.cleanunet_network import CleanunetNetwork
from trainers.trainable import Trainable


class CleanunetModel(Trainable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = CleanunetNetwork().to(device)
        self.optimizer = Adam(self.module.parameters(), 2e-4)

        self.mrstft = MrstftLoss.for_speech(device)

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
        speech: Tensor3d = self.module(reverb_batch.unsqueeze(1))
        return Tensor2d(speech.squeeze(1))

    def _calculate_losses(self, 
                          actual: Tensor2d,
                          predicted: Tensor2d) -> tuple[Tensor0d, dict[str, float]]:
        mrstft: MrstftLoss.Return = self.mrstft(actual, predicted)
        l1: torch.Tensor = torch.nn.functional.l1_loss(predicted, actual)
        total: Tensor0d = Tensor0d(0.5 * mrstft.total() + l1)
        return total, {
            "loss_total": float(total),
            "loss_mrstft_mag": float(mrstft.mag_loss),
            "loss_mrstft_sc": float(mrstft.sc_loss),
            "loss_l1": float(l1)
        }

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch)
        
        losses: dict[str, float]
        loss_total: Tensor0d
        loss_total, losses = self._calculate_losses(speech_batch, predicted)

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1e9)
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
        _, losses = self._calculate_losses(speech_batch, predicted)

        self.module.train()
        
        return losses["loss_total"], losses
