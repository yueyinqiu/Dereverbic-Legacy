# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

from typing import Any, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from models.fins_models.fins_network import FinsNetwork
from metrics.stft_losses.mrstft_loss import MrstftLoss
from trainers.trainable import Trainable

from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]


class FinsModel(Trainable):
    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__()
        self.device = device

        self.module = FinsNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), lr=0.000055, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.8
        )
        self.random = torch.Generator(device).manual_seed(seed)
        self.loss = MrstftLoss(device,                 
                               fft_sizes=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                               hop_sizes=[i * 16000 // 48000 for i in [32, 256, 1024, 4096]],
                               win_lengths=[i * 16000 // 48000 for i in [64, 512, 2048, 8192]],
                               window="hann_window")

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        scheduler: dict[str, Any]
        random: torch.Tensor

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "random": self.random.get_state()
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
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

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        losses: MrstftLoss.Return = self.loss(predicted, rir_batch)
        loss_total: Tensor0d = losses.total()

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), 5)
        self.optimizer.step()

        result: dict[str, float] = {
            "loss_total": float(loss_total),
            "loss_mag": float(losses.mag_loss),
            "loss_sc": float(losses.sc_loss),
            "lr": self.scheduler.get_last_lr()[0]
        }

        self.scheduler.step()
        return result

    def evaluate_on(self, reverb_batch: Tensor2d) -> Tensor2d:
        self.module.eval()
        predicted: Tensor2d = self._predict(reverb_batch, None, None)
        self.module.train()
        return predicted
    