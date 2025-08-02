import csv
from pathlib import Path
import sys
import csfile
from statictorch import Tensor2d
import torch
from torch.utils.data import DataLoader

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.l1_loss_metric import L1LossMetric
from metrics.metric import Metric
from metrics.mrstft_loss_metric import MrstftLossMetric
from metrics.pesq_metric import PesqMetric
from metrics.stoi_metric import StoiMetric
from models.cleanunet_models.cleanunet_model import CleanunetModel
from trainers.trainer import Trainer


def test(model: CleanunetModel, 
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
            predicted: Tensor2d = model.evaluate_on(batch.reverb)

            metric: str
            for metric in metrics:
                current: dict[str, float] = metrics[metric].append(batch.speech, predicted)
                submetric: str
                for submetric in current:
                    csv_print.writerow([batch_index, metric, submetric, current[submetric]])

        for metric in metrics:
            value: float
            for submetric, value in metrics[metric].result().items():
                csv_print.writerow(["all", metric, submetric, value])


def main():
    from exe.cleanunet.dereverb import test_cleanunet_config as config

    print("# Loading...")
    test(CleanunetModel(config.device), 
         CheckpointsDirectory(config.checkpoints_directory), 
         ValidationOrTestDataset(config.test_list, config.device).get_data_loader(32),
         {
             "mrstft": MrstftLossMetric.for_speech(config.device),
             "l1": L1LossMetric(config.device),
             "pesq": PesqMetric(16000),
             "stoi": StoiMetric(16000),
         })


if __name__ == "__main__":
    main()