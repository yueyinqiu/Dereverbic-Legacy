from typing import Any, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from criterions.stft_losses.mrstft_loss import MrstftLoss
from models.fins_models.fins_network import FinsNetwork
from trainers.trainable import Trainable

from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]


class FinsModel(Trainable):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = FinsNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), lr=0.000055, weight_decay=1e-6)
        self.random = torch.Generator(device).manual_seed(seed)
        self.loss = MrstftLoss.for_rir(device)

        self.train_preparation: Tensor0d | None = None

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        random: torch.Tensor

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "random": self.random.get_state()
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.random.set_state(state["random"])

    def _predict(self, 
                 reverb_batch: Tensor2d, 
                 stochastic_noise_batch: Tensor3d | None, 
                 noise_condition: Tensor2d | None) \
                    -> Tensor2d:
        b: int = reverb_batch.size()[0]

        if stochastic_noise_batch is None:
            stochastic_noise_batch = Tensor3d(
                torch.randn((b, 1, self.module.rir_length), 
                            generator=self.random,
                            device=self.device))
            stochastic_noise_batch = Tensor3d(
                stochastic_noise_batch.repeat(1, self.module.num_filters, 1))

        if noise_condition is None:
            noise_condition = Tensor2d(
                torch.randn((b, self.module.noise_condition_length), 
                            generator=self.random, 
                            device=self.device))
        predicted: Tensor3d = self.module(
            reverb_batch.unsqueeze(1), 
            stochastic_noise_batch, 
            noise_condition)
        return Tensor2d(predicted.squeeze(1))

    def _calculate_losses(self, 
                          actual: Tensor2d,
                          predicted: Tensor2d) -> tuple[Tensor0d, dict[str, float]]:
        losses: MrstftLoss.Return = self.loss(actual, predicted)
        loss_total: Tensor0d = losses.total()
        return loss_total, {
            "loss_total": float(loss_total),
            "loss_mag": float(losses.mag_loss),
            "loss_sc": float(losses.sc_loss)
        }

    def prepare_train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch, None, None)

        result: dict[str, float]
        self.train_preparation, result = self._calculate_losses(rir_batch, predicted)

        return result

    def train_prepared(self):
        assert self.train_preparation is not None
        self.optimizer.zero_grad()
        self.train_preparation.backward()
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5)
        self.optimizer.step()

    def evaluate_on(self, reverb_batch: Tensor2d) -> Tensor2d:
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        self.module.train()
        return predicted
    
    def validate_on(self,
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        self.module.eval()

        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        losses: dict[str, float]
        _, losses = self._calculate_losses(rir_batch, predicted)

        self.module.train()
        
        return losses["loss_total"], losses
