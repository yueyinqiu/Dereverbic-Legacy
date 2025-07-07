from statictorch import Tensor1d

from audio_processors.rir_acoustic_features import RirAcousticFeatures2d
from metrics.metric import Metric


class RirDirectToReverberantEnergyRatioMetrics(Metric[RirAcousticFeatures2d]):
    def __init__(self, 
                 metrics_on_direct_to_reverberant_energy_ratio: dict[str, Metric[Tensor1d]]):
        self._metrics = metrics_on_direct_to_reverberant_energy_ratio
    
    def append(self, actual: RirAcousticFeatures2d, predicted: RirAcousticFeatures2d) -> dict[str, float]:
        actual_value: Tensor1d = actual.direct_to_reverberant_energy_ratio_db()
        predicted_value: Tensor1d = predicted.direct_to_reverberant_energy_ratio_db()

        result: dict[str, float] = {}

        key_metric: str
        for key_metric in self._metrics:
            values: dict[str, float] = self._metrics[key_metric].append(actual_value, predicted_value)
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
