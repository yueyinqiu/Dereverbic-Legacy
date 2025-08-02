import csv
from pathlib import Path
import sys
import csfile
from statictorch import Tensor2d
import torch
from torch.utils.data import DataLoader

from audio_processors.rir_acoustic_features import RirAcousticFeatures2d
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.bias_metric import BiasMetric
from metrics.l1_loss_metric import L1LossMetric
from metrics.metric import Metric
from metrics.mrstft_loss_metric import MrstftLossMetric
from metrics.pearson_correlation_metric import PearsonCorrelationMetric
from metrics.pesq_metric import PesqMetric
from metrics.rir_direct_to_reverberant_energy_ratio_metrics import RirDirectToReverberantEnergyRatioMetrics
from metrics.rir_reverberation_time_metrics import RirReverberationTimeMetrics
from metrics.stoi_metric import StoiMetric
from models.cleanunet_models.cleanunet_two_stage_model import CleanUNetTwoStageModel
from trainers.trainer import Trainer


def test(model: CleanUNetTwoStageModel, 
         checkpoints: CheckpointsDirectory, 
         data: DataLoader, 
         rir_metrics: dict[str, Metric[Tensor2d]],
         rir_feature_metrics: dict[str, Metric[RirAcousticFeatures2d]],
         speech_metrics: dict[str, Metric[Tensor2d]]):
    with torch.no_grad():
        print(f"# Batch count: {data.__len__()}")

        rank_file: Path = checkpoints.get_path(None) / "validation_rank.txt"
        if rank_file.exists():
            epoch: int = int(csfile.read_all_lines(rank_file)[0])
            path: Path = checkpoints.get_path(epoch)
            print(f"# Rank file found. The best checkpoint {epoch} will be used.")
        else:
            latest: EpochAndPath | None = checkpoints.get_latest()
            if not latest:
                raise FileNotFoundError("Failed to find any checkpoint in the checkpoints directory.")
            epoch, path = latest
            print(f"# Failed to find the rank file. The latest checkpoint {epoch} will be used.")

        csv_print: CsvWriter = csv.writer(sys.stdout)
        csv_print.writerow(["batch", "task", "metric", "submetric", "value"])

        Trainer.load_model(model, path)

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: CleanUNetTwoStageModel.Prediction = model.evaluate_on(batch.reverb)

            metric: str
            for metric in rir_metrics:
                current: dict[str, float] = rir_metrics[metric].append(batch.rir, predicted.rir)
                submetric: str
                for submetric in current:
                    csv_print.writerow([batch_index, "rir", metric, submetric, current[submetric]])

            actual_features: RirAcousticFeatures2d = RirAcousticFeatures2d(batch.rir)
            predicted_features: RirAcousticFeatures2d = RirAcousticFeatures2d(predicted.rir)
            for metric in rir_feature_metrics:
                current = rir_feature_metrics[metric].append(actual_features, predicted_features)
                for submetric in current:
                    csv_print.writerow([batch_index, "rir", metric, submetric, current[submetric]])

            for metric in speech_metrics:
                current = speech_metrics[metric].append(batch.speech, predicted.speech)
                for submetric in current:
                    csv_print.writerow([batch_index, "speech", metric, submetric, current[submetric]])

        for metric in rir_metrics:
            value: float
            for submetric, value in rir_metrics[metric].result().items():
                csv_print.writerow(["all", "rir", metric, submetric, value])

        for metric in rir_feature_metrics:
            for submetric, value in rir_feature_metrics[metric].result().items():
                csv_print.writerow(["all", "rir", metric, submetric, value])

        for metric in speech_metrics:
            for submetric, value in speech_metrics[metric].result().items():
                csv_print.writerow(["all", "speech", metric, submetric, value])


def main():
    from exe.cleanunet.two_stage import test_cleanunet_two_stage_config as config

    print("# Loading...")
    test(CleanUNetTwoStageModel(config.device), 
         CheckpointsDirectory(config.checkpoints_directory), 
         ValidationOrTestDataset(config.test_list, config.device).get_data_loader(32),
         {
             "mrstft": MrstftLossMetric.for_rir(config.device),
             "l1": L1LossMetric(config.device)
         },
         {
             "rt60": RirReverberationTimeMetrics(30, 16000, {
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(config.device),
                 "pearson": PearsonCorrelationMetric()
             }),
             "drr": RirDirectToReverberantEnergyRatioMetrics({
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(config.device),
                 "pearson": PearsonCorrelationMetric()
             }),
         },
         {
             "mrstft": MrstftLossMetric.for_speech(config.device),
             "l1": L1LossMetric(config.device),
             "pesq": PesqMetric(16000),
             "stoi": StoiMetric(16000),
         })


if __name__ == "__main__":
    main()