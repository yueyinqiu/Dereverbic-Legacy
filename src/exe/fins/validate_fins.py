import csv
from pathlib import Path
from random import Random
import sys

import csfile
from statictorch import Tensor2d
import torch
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.kahan_accumulator import KahanAccumulator
from metrics.stft_losses.mrstft_loss import MrstftLoss
from models.fins_models.fins_model import FinsModel
from trainers.trainer import Trainer

from torch.utils.data import DataLoader


def main():
    from exe.fins import validate_fins_config as config
    
    print("# Loading...")
    with torch.no_grad():
        random: Random = Random(config.random_seed)
        checkpoints: CheckpointsDirectory = CheckpointsDirectory(config.checkpoints_directory)
        model: FinsModel = FinsModel(config.device, random.randint(0, 1000))
        data: DataLoader = ValidationOrTestDataset(config.validation_list, 
                                                   config.device).get_data_loader(32)
        mrstft: MrstftLoss = model.loss
        print(f"# Batch count: {data.__len__()}")

        scores: dict[int, float] = {}

        csv_print: CsvWriter = csv.writer(sys.stdout)
        csv_print.writerow(["epoch", "batch", "metric", "value"])
        epoch: int
        path: Path
        for epoch, path in checkpoints.get_all():
            if epoch < config.start_checkpoint:
                continue
            
            Trainer.load_model(model, path)

            mrstft_total_accumulator: KahanAccumulator = KahanAccumulator()
            mrstft_sc_accumulator: KahanAccumulator = KahanAccumulator()
            mrstft_mag_accumulator: KahanAccumulator = KahanAccumulator()

            batch_index: int
            batch: DataBatch
            for batch_index, batch in enumerate(data):
                predicted: Tensor2d = model.evaluate_on(batch.reverb)
                mrstft_value: MrstftLoss.Return = mrstft(predicted, batch.rir)
                
                mrstft_total: float = float(mrstft_value.total())
                mrstft_sc: float = float(mrstft_value.sc_loss)
                mrstft_mag: float = float(mrstft_value.mag_loss)

                mrstft_total_accumulator.add(mrstft_total)
                mrstft_sc_accumulator.add(mrstft_sc)
                mrstft_mag_accumulator.add(mrstft_mag)

                csv_print.writerow([epoch, batch_index, "mrstft_total", mrstft_total])
                csv_print.writerow([epoch, batch_index, "mrstft_sc", mrstft_sc])
                csv_print.writerow([epoch, batch_index, "mrstft_mag", mrstft_mag])

            score: float = mrstft_total_accumulator.value() / data.__len__()
            scores[epoch] = score

            csv_print.writerow([epoch, "average", "mrstft_total", 
                                score])
            csv_print.writerow([epoch, "average", "mrstft_sc", 
                                mrstft_sc_accumulator.value() / data.__len__()])
            csv_print.writerow([epoch, "average", "mrstft_mag", 
                                mrstft_mag_accumulator.value() / data.__len__()])

        csfile.write_all_lines(checkpoints.get_path(None) / "validation_rank.txt", 
                            [str(key) for key in sorted(scores.keys(), key=lambda key: scores[key])])


if __name__ == "__main__":
    main()
