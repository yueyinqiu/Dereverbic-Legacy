from typing import Any, NamedTuple, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]

from criterions.rir_energy_decay_loss.rir_energy_decay_loss import RirEnergyDecayLoss
from criterions.stft_losses.mrstft_loss import MrstftLoss
from models.ricbe_models.networks.tdunet_dereverb_network import TdunetDereverbNetwork
from models.ricbe_models.networks.dereverbic_network import DereverbicNetwork
from models.ricbe_models.networks.tdunet_ric_network import TdunetRicNetwork
from trainers.trainable import Trainable


class DereverbicModel(Trainable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = DereverbicNetwork(TdunetDereverbNetwork(), TdunetRicNetwork(False, False)).to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)

        self.mrstft = MrstftLoss.for_rir(device)
        self.l1 = torch.nn.L1Loss()
        self.energy_decay = RirEnergyDecayLoss()
        self.speech_mrstft = MrstftLoss.for_speech(device)
        
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

    class Prediction(NamedTuple):
        speech: Tensor2d
        rir: Tensor2d

    def _predict(self,
                 reverb_batch: Tensor2d) -> Prediction:
        speech: Tensor3d
        rir: Tensor3d
        speech, rir = self.module(reverb_batch.unsqueeze(1))
        return DereverbicModel.Prediction(Tensor2d(speech.squeeze(1)), 
                                         Tensor2d(rir.squeeze(1)))

    def _calculate_losses(self, 
                          actual_rir: Tensor2d,
                          actual_speech: Tensor2d,
                          predicted: Prediction) -> tuple[Tensor0d, dict[str, float]]:
        mrstft: MrstftLoss.Return = self.mrstft(actual_rir, predicted.rir)
        l1: torch.Tensor = self.l1(actual_rir, predicted.rir)
        energy: torch.Tensor = self.energy_decay(actual_rir, predicted.rir)

        speech_mrstft: MrstftLoss.Return = self.speech_mrstft(actual_speech, predicted.speech)

        total: Tensor0d = Tensor0d(mrstft.total() + l1 + energy + speech_mrstft.total())
        return total, {
            "loss_total": float(total),
            "loss_rir_mrstft_mag": float(mrstft.mag_loss),
            "loss_rir_mrstft_sc": float(mrstft.sc_loss),
            "loss_rir_l1": float(l1),
            "loss_rir_energy_decay": float(energy),
            "loss_speech_mrstft_mag": float(speech_mrstft.mag_loss),
            "loss_speech_mrstft_sc": float(speech_mrstft.sc_loss),
        }

    def prepare_train_on(self, 
                         reverb_batch: Tensor2d, 
                         rir_batch: Tensor2d,
                         speech_batch: Tensor2d) -> dict[str, float]:
        predicted: DereverbicModel.Prediction = self._predict(reverb_batch)
        
        losses: dict[str, float]
        self.train_preparation, losses = self._calculate_losses(rir_batch, 
                                                                speech_batch, 
                                                                predicted)

        return losses

    def train_prepared(self):
        assert self.train_preparation is not None
        self.optimizer.zero_grad()
        self.train_preparation.backward()
        self.optimizer.step()

    def evaluate_on(self, 
                    reverb_batch: Tensor2d) -> Prediction:
        self.module.eval()
        predicted: DereverbicModel.Prediction = self._predict(reverb_batch)
        self.module.train()
        return predicted
    
    def validate_on(self,
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        self.module.eval()

        predicted: DereverbicModel.Prediction = self._predict(reverb_batch)
        losses: dict[str, float]
        _, losses = self._calculate_losses(rir_batch, speech_batch, predicted)

        self.module.train()
        return losses["loss_total"], losses
