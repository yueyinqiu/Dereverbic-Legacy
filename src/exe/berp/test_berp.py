import csv
import io
from pathlib import Path
import sys
from turtle import st
from typing import Iterable, TypedDict
import csfile
from networkx import volume
from statictorch import Tensor1d, Tensor2d, anify
import torch

from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath
from inputs_and_outputs.csv_accessors.csv_reader import CsvReader
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.bias_metric import BiasMetric
from metrics.l1_loss_metric import L1LossMetric
from metrics.metric import Metric
from metrics.mrstft_loss_metric import MrstftLossMetric
from metrics.pearson_correlation_metric import PearsonCorrelationMetric
from metrics.rir_reverberation_time_metrics import RirReverberationTimeMetrics
from models.berp_models.berp_hybrid_model import BerpHybridModel
from trainers.trainer import Trainer

class _TestDatasetWithVolume(torch.utils.data.Dataset):
    @staticmethod
    def load_reverb_path_to_rir_path_map(reverb_contents_csv_path: Path):
        print("# Mapping Reverb...")

        result: dict[str, str] = {}

        csv_file: io.TextIOWrapper
        with open(reverb_contents_csv_path, newline="") as csv_file:
            csv_reader: CsvReader = csv.reader(csv_file)

            row_str: list[str]
            for row_str in csv_reader:
                assert tuple(row_str)[0] == "Reverb"
                assert tuple(row_str)[1] == "Rir"
                assert tuple(row_str)[2] == "Speech"
                break

            for row_str in csv_reader:
                reverb_path: str = row_str[0]
                rir_path: str = row_str[1]
                result[reverb_path] = rir_path

        return result

    @staticmethod
    def load_tensor_path_to_raw_rir_id_map(rir_contents_csv_path: Path):
        print("# Mapping Rir...")

        result: dict[str, str] = {}

        csv_file: io.TextIOWrapper
        with open(rir_contents_csv_path, newline="") as csv_file:
            csv_reader: CsvReader = csv.reader(csv_file)

            row_str: list[str]
            for row_str in csv_reader:
                assert tuple(row_str)[0] == "Tensor"
                assert tuple(row_str)[1] == "Original Audio"
                assert tuple(row_str)[2] == "Original Channel"
                break

            for row_str in csv_reader:
                tensor_path: str = row_str[0]
                audio_path: str = row_str[1]
                result[tensor_path] = Path(audio_path).stem

        return result

    @staticmethod
    def load_rir_id_to_room_volume_map(fold02d_csv_paths: Iterable[Path]):
        print("# Loading Room Volumes...")

        result: dict[str, float] = {}

        fold02d_csv_path: Path
        for fold02d_csv_path in fold02d_csv_paths:
            csv_file: io.TextIOWrapper
            with open(fold02d_csv_path, newline="") as csv_file:
                csv_reader: CsvReader = csv.reader(csv_file)

                row_str: list[str]
                for row_str in csv_reader:
                    assert ",".join(row_str) == \
                        "id,Lx,Ly,Lz,alpha,c,m1x,m1y,m1z,m2x,m2y,m2z,s1x,s1y,s1z,s2x,s2y,s2z,s3x,s3y,s3z,s4x,s4y,s4z"
                    break

                for row_str in csv_reader:
                    id: str = row_str[0]
                    Lx: str = row_str[1]
                    Ly: str = row_str[2]
                    Lz: str = row_str[3]
                    result[id] = float(Lx) * float(Ly) * float(Lz)

        return result

    def __init__(self, 
                 reverb_map: Path, rir_map: Path, rir_information: Iterable[Path],
                 data_list: Path, 
                 device: torch.device):
        self._paths = csfile.read_all_lines(data_list)
        self._device = device

        self._rir_volume: dict[str, float] = {}
        
        reverb_to_rir_map: dict[str, str] = _TestDatasetWithVolume.load_reverb_path_to_rir_path_map(reverb_map)
        id_map: dict[str, str] = _TestDatasetWithVolume.load_tensor_path_to_raw_rir_id_map(rir_map)
        volume_map: dict[str, float] = _TestDatasetWithVolume.load_rir_id_to_room_volume_map(rir_information)

        rir_path: str
        for rir_path in self._paths:
            volume: float = volume_map[id_map[reverb_to_rir_map[rir_path]]]
            self._rir_volume[rir_path] = volume

    def __len__(self):
        return self._paths.__len__()

    class DatasetItem(TypedDict):
        rir: Tensor1d
        speech: Tensor1d
        reverb: Tensor1d

    def __getitem__(self, i: int) -> tuple[DatasetItem, float]:
        path: str = self._paths[i]
        return torch.load(path, weights_only=True, map_location=self._device), self._rir_volume[path]

    def get_data_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        def collate(data: list[tuple[ValidationOrTestDataset.DatasetItem, float]]) -> tuple[DataBatch, Tensor1d]:
            rirs: list[Tensor1d] = []
            speeches: list[Tensor1d] = []
            reverbs: list[Tensor1d] = []
            volumes: list[float] = []

            item: ValidationOrTestDataset.DatasetItem
            volume: float
            for item, volume in data:
                rirs.append(item["rir"])
                speeches.append(item["speech"])
                reverbs.append(item["reverb"])
                volumes.append(volume)
            
            result: DataBatch = DataBatch(Tensor2d(torch.stack(anify(rirs))), 
                                          Tensor2d(torch.stack(anify(speeches))), 
                                          Tensor2d(torch.stack(anify(reverbs))))
            return result, Tensor1d(torch.tensor(volumes, device=self._device))

        return torch.utils.data.DataLoader(self, batch_size, False, collate_fn=collate)


def test(model: BerpHybridModel, 
         checkpoints: CheckpointsDirectory,
         data: torch.utils.data.DataLoader, 
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
        batch: tuple[DataBatch, Tensor1d]
        for batch_index, batch in enumerate(data):
            predicted: Tensor2d = model.evaluate_rir_on(batch[0].reverb, batch[1])

            metric: str
            for metric in metrics:
                current: dict[str, float] = metrics[metric].append(batch[0].rir, predicted)
                submetric: str
                for submetric in current:
                    csv_print.writerow([batch_index, metric, submetric, current[submetric]])

        for metric in metrics:
            value: float
            for submetric, value in metrics[metric].result().items():
                csv_print.writerow(["all", metric, submetric, value])


def main():
    from exe.berp import test_berp_config as config
    
    data: _TestDatasetWithVolume = _TestDatasetWithVolume(config.reverb_map, 
                                                          config.rir_map, 
                                                          config.rir_information, 
                                                          config.test_list,
                                                          config.device)
    test(BerpHybridModel(config.device),
         CheckpointsDirectory(config.checkpoints_directory),
         data.get_data_loader(32),
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