# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from typing import Any, NamedTuple, TypedDict
from statictorch import Tensor0d, Tensor2d, Tensor3d
import torch
from models.ricbe_models.ricbe_network import RicbeNetwork
from metrics.stft_losses.mrstft_loss import MrstftLoss
from trainers.rir_blind_estimation_model import Trainable
from torch.optim import AdamW  # pyright: ignore [reportPrivateImportUsage]


class RicbeModel(Trainable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = RicbeNetwork().to(device)
        self.optimizer = AdamW(self.module.parameters(), 0.0001)

        self.speech_mrstft = MrstftLoss(device, 
                                        fft_sizes=[256, 512, 1024, 2048], 
                                        hop_sizes=[64, 128, 256, 512], 
                                        win_lengths=[256, 512, 1024, 2048],
                                        window="hann_window")
        self.rir_mrstft = MrstftLoss(device, 
                                     fft_sizes=[32, 256, 1024, 4096],
                                     hop_sizes=[16, 128, 512, 2048],
                                     win_lengths=[32, 256, 1024, 4096], 
                                     window="hann_window")
        self.l1 = torch.nn.L1Loss().to(device)

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
        rir: Tensor2d
        speech: Tensor2d

    def _predict(self,
                 reverb_batch: Tensor2d) -> Prediction:
        rir: Tensor3d
        speech: Tensor3d
        rir, speech = self.module(reverb_batch.unsqueeze(1))
        return RicbeModel.Prediction(Tensor2d(rir.squeeze(1)), 
                                     Tensor2d(speech.squeeze(1)))

    def train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)

        rir_l1: Tensor0d = self.l1(predicted.rir, rir_batch)
        rir_mrstft: Tensor0d = self.rir_mrstft(predicted.rir, rir_batch).total()
        speech_l1: Tensor0d = self.l1(predicted.speech, speech_batch)
        speech_mrstft: Tensor0d = self.speech_mrstft(predicted.speech, speech_batch).total()
        
        loss_rir: Tensor0d = Tensor0d(rir_l1 + rir_mrstft)
        loss_speech: Tensor0d = Tensor0d(speech_l1 + speech_mrstft)
        loss_total: Tensor0d = Tensor0d(loss_rir + loss_speech)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            "loss_total": float(loss_total),
            "loss_rir": float(loss_rir),
            "loss_speech": float(loss_speech)
        }

    def evaluate_on(self, reverb_batch: Tensor2d):
        self.module.eval()
        predicted: RicbeModel.Prediction = self._predict(reverb_batch)
        self.module.train()
        return predicted
    