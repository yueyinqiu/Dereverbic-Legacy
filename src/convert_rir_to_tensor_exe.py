import csv
import io
from pathlib import Path
from random import Random
import csdir
from statictorch import Tensor1d, Tensor2d
import torch

from basic_utilities.string_random import StringRandom
import convert_rir_to_tensor_config
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.tensor_audios.tensor_audios import TensorAudios


def _save_tensor(audio: Tensor1d,
                 file_name_without_suffix: str,
                 directory: Path):
    directory = directory / file_name_without_suffix[0] / file_name_without_suffix[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: Path = directory / f"{file_name_without_suffix}.wav.pt"
    torch.save(audio, tensor_file)
    return tensor_file


def main():
    rand: Random = Random(convert_rir_to_tensor_config.random_seed)
    string_random: StringRandom = StringRandom(rand, 16)

    print("Sorting files ...")
    inputs: list[Path] = sorted(convert_rir_to_tensor_config.inputs)

    csdir.create_directory(convert_rir_to_tensor_config.output_directory)
    contents_file: io.TextIOWrapper
    with open(convert_rir_to_tensor_config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer: CsvWriter = csv.writer(contents_file)
        contents_writer.writerow(["Tensor", "Original Audio", "Original Channel"])

        path: Path
        for path in inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio: Tensor2d = TensorAudios.load_audio(path, 16000, "as_many")
            
            i: int
            for i in range(audio.shape[0]):
                tensor_file: Path = _save_tensor(Tensor1d(audio[i]), 
                                                 string_random.next(), 
                                                 convert_rir_to_tensor_config.output_directory)
                contents_writer.writerow([str(tensor_file), str(path), str(i)])
                contents_file.flush()
                
    print(f"Completed.")
    
main()