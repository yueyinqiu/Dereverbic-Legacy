from typing import Any, Callable, NamedTuple, TypedDict
import einops
import nnAudio
import nnAudio.features
from statictorch import Tensor0d, Tensor1d, Tensor2d, anify
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from models.berp_models.networks.berp_joint_estimator import BerpJointEstimator
from models.berp_models.networks.berp_rir_utilities import BerpRirUtilities
from models.berp_models.networks.berp_sparse_stochastic_ir import BerpSparseStochasticIr
from trainers.trainable import Trainable

from torch.optim import RAdam  # pyright: ignore [reportPrivateImportUsage]

class BerpHybridModel(Trainable):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.module = BerpJointEstimator().to(device)
        self.optimizer = RAdam(self.module.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-6)
        self.loss = torch.nn.SmoothL1Loss().to(device)

        self.t_h_accumulator = KahanAccumulator()
        self.trained_data_count = 0
        
        self.mfcc = nnAudio.features.mel.MFCC(
            sr=16000, n_mfcc=128, n_fft=1024, hop_length=256,
            n_mels=128, power=1.0, verbose=False
        ).to(device)
        
        self.train_preparation: tuple[Tensor0d, Tensor1d] | None = None

    class StateDict(TypedDict):
        model: dict[str, Any]
        optimizer: dict[str, Any]
        t_h_accumulator: KahanAccumulator.StateDict
        trained_data_count: int

    def get_state(self) -> StateDict:
        return {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "t_h_accumulator": self.t_h_accumulator.get_state(),
            "trained_data_count": self.trained_data_count
        }

    def set_state(self, state: StateDict):
        self.module.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.t_h_accumulator.set_state(state["t_h_accumulator"])
        self.trained_data_count = state["trained_data_count"]

    class Prediction(NamedTuple):
        t_h: Tensor1d
        t_t: Tensor1d

    class _Prediction2d(NamedTuple):
        t_h: Tensor2d
        t_t: Tensor2d

    def _predict(self, reverb_batch: Tensor2d):
        mfcc: torch.Tensor = self.mfcc(reverb_batch)
        t_h: Tensor2d
        t_t: Tensor2d
        t_h, t_t = self.module(mfcc)
        return BerpHybridModel._Prediction2d(t_h, t_t)

    @staticmethod
    def _execute_on_each(f: Callable[[Tensor1d], Tensor0d], batch_input: Tensor2d) -> Tensor1d:
        result: list[torch.Tensor] = []
        element: Tensor1d
        for element in anify(batch_input):
            result.append(f(element))
        return anify(torch.stack(result))
        
    def _calculate_losses(self, 
                          actual: Tensor2d,
                          predicted: _Prediction2d) -> tuple[tuple[Tensor0d, Tensor1d], dict[str, float]]:
        actual_t_h_1d: Tensor1d = BerpHybridModel._execute_on_each(BerpRirUtilities.get_t_h, actual)
        actual_t_h: torch.Tensor = einops.repeat(actual_t_h_1d, "B -> B T", T=predicted.t_h.shape[1])
        loss_t_h: torch.Tensor = self.loss(actual_t_h, predicted.t_h)

        actual_t_t: torch.Tensor = BerpHybridModel._execute_on_each(BerpRirUtilities.get_t_t, actual)
        actual_t_t = einops.repeat(actual_t_t, "B -> B T", T=predicted.t_t.shape[1])
        loss_t_t: torch.Tensor = self.loss(actual_t_t, predicted.t_t)

        loss_total: Tensor0d = Tensor0d(loss_t_h + loss_t_t)
        return (loss_total, actual_t_h_1d), {
            "loss_total": float(loss_total),
            "loss_t_h": float(loss_t_h),
            "loss_t_t": float(loss_t_t)
        }

    def prepare_train_on(self, 
                 reverb_batch: Tensor2d, 
                 rir_batch: Tensor2d, 
                 speech_batch: Tensor2d) -> dict[str, float]:
        predicted: BerpHybridModel._Prediction2d = self._predict(reverb_batch)

        result: dict[str, float]
        self.train_preparation, result = self._calculate_losses(rir_batch, predicted)

        return result

    def train_prepared(self):
        assert self.train_preparation is not None
        self.optimizer.zero_grad()
        self.train_preparation[0].backward()
        self.optimizer.step()

        t_h: torch.Tensor
        for t_h in self.train_preparation[1]:
            self.t_h_accumulator.add(float(t_h))
            self.trained_data_count += 1

    def evaluate_on(self, reverb_batch: Tensor2d) -> Prediction:
        self.module.eval()
        predicted: BerpHybridModel._Prediction2d = self._predict(reverb_batch)
        self.module.train()
        return BerpHybridModel.Prediction(Tensor1d(predicted.t_h.mean(-1)),
                                          Tensor1d(predicted.t_t.mean(-1)))
    
    def evaluate_rir_on(self, 
                        reverb_batch: Tensor2d, volume_batch: Tensor1d,
                        ssir_seed: int = 0) -> Tensor2d:
        t_h_batch: Tensor1d
        t_t_batch: Tensor1d
        t_h_batch, t_t_batch = self.evaluate_on(reverb_batch)

        mu_t_h: float = self.t_h_accumulator.value() / self.trained_data_count
        ssir: BerpSparseStochasticIr = BerpSparseStochasticIr(mu_t_h, ssir_seed)
        
        result: list[torch.Tensor] = []
        t_h: torch.Tensor
        t_t: torch.Tensor
        volume: torch.Tensor
        for t_h, t_t, volume in zip(t_h_batch, t_t_batch, volume_batch, strict=True):
            result.append(ssir(float(t_h), float(t_t), float(volume)))
        
        return Tensor2d(torch.stack(result))
    
    def validate_on(self,
                    reverb_batch: Tensor2d, 
                    rir_batch: Tensor2d, 
                    speech_batch: Tensor2d) -> tuple[float, dict[str, float]]:
        self.module.eval()

        predicted: BerpHybridModel._Prediction2d = self._predict(reverb_batch)
        losses: dict[str, float]
        _, losses = self._calculate_losses(rir_batch, predicted)

        self.module.train()
        
        return losses["loss_total"], losses
