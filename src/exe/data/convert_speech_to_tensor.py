import csv
import io
from pathlib import Path
from random import Random
import csdir
import librosa
import numpy
from statictorch import Tensor1d, Tensor2d
import torch

from basic_utilities.string_random import StringRandom
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
    from exe.data import convert_speech_to_tensor_config as config

    rand: Random = Random(config.random_seed)
    string_random: StringRandom = StringRandom(rand, 16)
    
    print("Sorting files ...")
    inputs: list[Path] = sorted(config.inputs)

    csdir.create_directory(config.output_directory)
    contents_file: io.TextIOWrapper
    with open(config.output_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as contents_file:
        contents_writer: CsvWriter = csv.writer(contents_file)
        contents_writer.writerow(["Tensor", "Original Audio"])

        path: Path
        for path in inputs:
            path = path.absolute()
            print(f"Dealing with {path} ...")

            audio: Tensor2d = TensorAudios.load_audio(path, 16000, "as_mono")
            channel: numpy.ndarray = audio[0, :].numpy()
            channel, _ = librosa.effects.trim(channel,
                                              top_db=60, 
                                              frame_length=2048, 
                                              hop_length=512)
            
            tensor: Tensor1d = Tensor1d(torch.tensor(channel, dtype=torch.float))
            start: int = 16000 // 5
            while start + 5 * 16000 < tensor.__len__() - 16000 // 5:
                tensor_file: Path = _save_tensor(Tensor1d(tensor[start:(start + 5 * 16000)]), 
                                                 string_random.next(), 
                                                 config.output_directory)
                contents_writer.writerow([str(tensor_file), str(path)])
                contents_file.flush()
                start += 5 * 16000
                
    print(f"Completed.")
    
main()