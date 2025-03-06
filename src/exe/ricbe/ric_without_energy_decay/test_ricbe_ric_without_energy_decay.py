import csv
from pathlib import Path
from random import Random
import sys
from typing import Callable
import csfile
from statictorch import Tensor1d, Tensor2d
import torch
from torch.utils.data import DataLoader

from audio_processors.rir_acoustic_features import RirAcousticFeatures
from basic_utilities.kahan_accumulator import KahanAccumulator
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
from metrics.rir_reverberation_time_metrics import RirReverberationTimeMetrics
from models.cleanunet_models.cleanunet_model import CleanunetModel
from models.fins_models.fins_model import FinsModel
from models.ricbe_models.ricbe_ric_model import RicbeRicModel
from trainers.trainer import Trainer


def test(model: RicbeRicModel, 
         checkpoints: CheckpointsDirectory, 
         data: DataLoader, 
         metrics: dict[str, Metric[Tensor2d]]):
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
        csv_print.writerow(["batch", "metric", "submetric", "value"])

        Trainer.load_model(model, path)

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: Tensor2d = model.evaluate_on(batch.reverb, batch.speech)

            metric: str
            for metric in metrics:
                current: dict[str, float] = metrics[metric].append(batch.rir, predicted)
                submetric: str
                for submetric in current:
                    csv_print.writerow([batch_index, metric, submetric, current[submetric]])

        for metric in metrics:
            value: float
            for submetric, value in metrics[metric].result().items():
                csv_print.writerow(["all", metric, submetric, value])


def main():
    from exe.ricbe.ric_without_energy_decay import test_ricbe_ric_without_energy_decay_config as config

    print("# Loading...")
    test(RicbeRicModel(config.device), 
         CheckpointsDirectory(config.checkpoints_directory), 
         ValidationOrTestDataset(config.test_list, config.device).get_data_loader(32),
         {
             "mrstft": MrstftLossMetric.for_rir(config.device),
             "l1": L1LossMetric(config.device),
             "rt30": RirReverberationTimeMetrics(30, 16000, {
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(config.device),
                 "pearson": PearsonCorrelationMetric()
             })
         })


if __name__ == "__main__":
    main()