import csv
from pathlib import Path
from random import Random
import sys
from typing import Callable

import csfile
from statictorch import Tensor1d, Tensor2d
import torch
from audio_processors.rir_acoustic_features import RirAcousticFeatures
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.kahan_accumulator import KahanAccumulator
from metrics.stft_losses.mrstft_loss import MrstftLoss

from torch.utils.data import DataLoader

from models.fins_models.fins_model import FinsModel
import test_fins_config
from trainers.trainer import Trainer


def test(model: FinsModel, 
         checkpoints: CheckpointsDirectory, 
         data: DataLoader, 
         criterions: list[Callable[[DataBatch, Tensor2d], dict[str, float]]]):
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
        csv_print.writerow(["batch", "metric", "value"])

        Trainer.load_model(model, path)

        accumulators: dict[str, KahanAccumulator] | None = None

        batch_index: int
        batch: DataBatch
        for batch_index, batch in enumerate(data):
            predicted: Tensor2d = model.evaluate_on(batch.reverb)

            losses: dict[str, float] = {}
            criterion: Callable[[DataBatch, Tensor2d], dict[str, float]]
            for criterion in criterions:
                losses.update(criterion(batch, predicted))

            if accumulators is None:
                accumulators = {key: KahanAccumulator() for key in losses}
            
            key: str
            for key in losses:
                csv_print.writerow([batch_index, key, losses[key]])
                accumulators[key].add(losses[key])

        assert accumulators is not None
        for key in accumulators:
            csv_print.writerow(["average", key, accumulators[key].value() / len(data)])


def main():
    print("# Loading...")
    mrstft_rir: MrstftLoss = MrstftLoss(test_fins_config.device, 
                                        fft_sizes=[32, 256, 1024, 4096],
                                        hop_sizes=[16, 128, 512, 2048],
                                        win_lengths=[32, 256, 1024, 4096], 
                                        window="hann_window")
                                     
    def criterion_rir_mrstft(actual: DataBatch, 
                             predicted: Tensor2d) -> dict[str, float]:
        values: MrstftLoss.Return = mrstft_rir(actual.rir, predicted)
        return {
            "mrstft": float(values.total()),
            "mrstft_mag": float(values.mag_loss),
            "mrstft_sc": float(values.sc_loss),
        }
    def criterion_reverberation_time(actual: DataBatch, 
                                     predicted: Tensor2d) -> dict[str, float]:
        a_edc: Tensor2d = RirAcousticFeatures.energy_decay_curve_decibel(actual.rir)
        a_rt30: Tensor1d = RirAcousticFeatures.get_reverberation_time_2d(a_edc, 
                                                                         sample_rate=16000)
        p_edc: Tensor2d = RirAcousticFeatures.energy_decay_curve_decibel(predicted.rir)
        p_rt30: Tensor1d = RirAcousticFeatures.get_reverberation_time_2d(p_edc, 
                                                                         sample_rate=16000)
        return {
            "rt30_bias": float(torch.mean(p_rt30 - a_rt30)),
            "rt30_mse": float(torch.nn.functional.mse_loss(p_rt30, a_rt30))
        }

    random: Random = Random(test_fins_config.random_seed)
    test(FinsModel(test_fins_config.device, random.randint(0, 1000)), 
         CheckpointsDirectory(test_fins_config.checkpoints_directory), 
         ValidationOrTestDataset(test_fins_config.test_list, test_fins_config.device).get_data_loader(32),
         [
             criterion_rir_mrstft, 
             criterion_reverberation_time
         ])


if __name__ == "__main__":
    main()