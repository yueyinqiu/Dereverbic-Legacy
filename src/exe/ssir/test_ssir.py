import csv
import io
from pathlib import Path
from random import Random
import sys
from typing import Iterable, NamedTuple
import csfile
from statictorch import Tensor2d
import torch

from basic_utilities.kahan_accumulator import KahanAccumulator
from inputs_and_outputs.csv_accessors.csv_reader import CsvReader
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.bias_metric import BiasMetric
from metrics.l1_loss_metric import L1LossMetric
from metrics.metric import Metric
from metrics.mrstft_loss_metric import MrstftLossMetric
from metrics.pearson_correlation_metric import PearsonCorrelationMetric
from metrics.rir_reverberation_time_metrics import RirReverberationTimeMetrics
from models.berp_models.berp_rir_utilities import BerpRirUtilities
from models.berp_models.berp_ssir_model import BerpSsirModel


class _PathInputModel:
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


    class RirInformation(NamedTuple):
        volume: float
        t_h: float
        t_t: float

    def __init__(self, reverb_map: Path, rir_map: Path, rir_information: Iterable[Path], test_list: list[str], seed: int) -> None:
        print("# Building Model...")

        cpu: torch.device = torch.device("cpu")

        reverb_to_rir_map: dict[str, str] = _PathInputModel.load_reverb_path_to_rir_path_map(reverb_map)
        id_map: dict[str, str] = _PathInputModel.load_tensor_path_to_raw_rir_id_map(rir_map)
        volume_map: dict[str, float] = _PathInputModel.load_rir_id_to_room_volume_map(rir_information)
        self.rir_information: dict[str, _PathInputModel.RirInformation] = {}

        t_h_accumulator: KahanAccumulator = KahanAccumulator()
        rir_path: str
        for rir_path in test_list:
            item: ValidationOrTestDataset.DatasetItem = torch.load(rir_path, 
                                                                weights_only=True, 
                                                                map_location=cpu)
            volume: float = volume_map[id_map[reverb_to_rir_map[rir_path]]]
            t_h: float = float(BerpRirUtilities.getTh(item["rir"], 16000))
            t_t: float = float(BerpRirUtilities.getTt(item["rir"], 16000))
            self.rir_information[rir_path] = _PathInputModel.RirInformation(volume, t_h, t_t)
            t_h_accumulator.add(t_h)
            print(f"# {rir_path}: volume={volume} t_h={t_h} t_t={t_t}")

        self.model: BerpSsirModel = BerpSsirModel(t_h_accumulator.value() / test_list.__len__(), seed)
    
    def __call__(self, path: str):
        information: _PathInputModel.RirInformation = self.rir_information[path]
        return self.model(information.t_h, information.t_t, information.volume)


def test(model: _PathInputModel, 
         data: list[str], 
         metrics: dict[str, Metric[Tensor2d]]):
    print(f"# Batch count: {data.__len__()}")

    csv_print: CsvWriter = csv.writer(sys.stdout)
    csv_print.writerow(["batch", "metric", "submetric", "value"])

    cpu: torch.device = torch.device("cpu")
    batch_index: int
    rir_path: str
    for batch_index, rir_path in enumerate(data):
        predicted: torch.Tensor = model(rir_path)
        actual: ValidationOrTestDataset.DatasetItem = torch.load(rir_path, 
                                                                 weights_only=True, 
                                                                 map_location=cpu)
        metric: str
        for metric in metrics:
            current: dict[str, float] = metrics[metric].append(Tensor2d(actual["rir"].unsqueeze(0)), 
                                                               Tensor2d(predicted.unsqueeze(0)))
            submetric: str
            for submetric in current:
                csv_print.writerow([batch_index, metric, submetric, current[submetric]])

    for metric in metrics:
        value: float
        for submetric, value in metrics[metric].result().items():
            csv_print.writerow(["all", metric, submetric, value])


def main():
    from exe.ssir import test_ssir_config as config
    random: Random = Random(config.random_seed)
    test_list: list[str] = csfile.read_all_lines(config.test_list)
    
    model: _PathInputModel = _PathInputModel(config.reverb_map,
                                             config.rir_map, 
                                             config.rir_information, 
                                             test_list, 
                                             random.randint(0, 1000))
    cpu: torch.device = torch.device("cpu")
    test(model, 
         test_list,
         {
             "mrstft": MrstftLossMetric.for_rir(cpu),
             "l1": L1LossMetric(cpu),
             "rt30": RirReverberationTimeMetrics(30, 16000, {
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(cpu),
                 "pearson": PearsonCorrelationMetric()
             })
         })


if __name__ == "__main__":
    main()