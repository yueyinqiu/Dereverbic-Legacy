from turtle import st
from typing import NamedTuple
from statictorch import Tensor0d, Tensor1d, Tensor2d
import torch

from audio_processors.rir_acoustic_features import RirAcousticFeatures
from basic_utilities.kahan_accumulator import KahanAccumulator
from criterions.stft_losses.mrstft_loss import MrstftLoss
from metrics.metric import Metric


class RirReverberationTimeMetrics(Metric[Tensor2d]):
    def __init__(self, 
                 decay_decibel: float,
                 sample_rate: int,
                 metrics_on_reverberation_time: dict[str, Metric[Tensor1d]]):
        self._decay_decibel = decay_decibel
        self._sample_rate = sample_rate
        self._metrics = metrics_on_reverberation_time
    
    def append(self, actual: Tensor2d, predicted: Tensor2d) -> dict[str, float]:
        actual = RirAcousticFeatures.energy_decay_curve_decibel(actual)
        actual_time: Tensor1d = RirAcousticFeatures.get_reverberation_time_2d(
            actual, self._decay_decibel, self._sample_rate)
        predicted = RirAcousticFeatures.energy_decay_curve_decibel(predicted)
        predicted_time: Tensor1d = RirAcousticFeatures.get_reverberation_time_2d(
            predicted, self._decay_decibel, self._sample_rate)

        result: dict[str, float] = {}

        key_metric: str
        for key_metric in self._metrics:
            values: dict[str, float] = self._metrics[key_metric].append(actual_time, predicted_time)
            key_value: str
            for key_value in values:
                result[f"{key_metric}_{key_value}"] = values[key_value]

        return result
        
    def result(self) -> dict[str, float]:
        result: dict[str, float] = {}

        key_metric: str
        for key_metric in self._metrics:
            values: dict[str, float] = self._metrics[key_metric].result()
            key_value: str
            for key_value in values:
                result[f"{key_metric}_{key_value}"] = values[key_value]

        return result
