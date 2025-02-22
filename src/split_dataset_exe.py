import csv
import io
from pathlib import Path
from random import Random

import csdir
import csfile
from statictorch import Tensor1d
import torch

from audio_processors.rir_convolution import RirConvolution
from basic_utilities.string_random import StringRandom
from inputs_and_outputs.csv_accessors.csv_reader import CsvReader
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
import split_dataset_config


def load_and_sort(contents: Path):
    tensor_files: list[str] = []

    csv_file: io.TextIOWrapper
    with open(contents, newline="") as csv_file:
        csv_reader: CsvReader = csv.reader(csv_file)

        row_str: list[str]
        for row_str in csv_reader:
            assert tuple(row_str)[0] == "Tensor"
            assert tuple(row_str)[1] == "Original Audio"
            break
        
        for row_str in csv_reader:
            tensor_files.append(row_str[0])

    tensor_files.sort()
    return tensor_files

def save_reverb(rir_path: str, 
                speech_path: str,
                name_generator: StringRandom,
                directory: Path):
    rir: Tensor1d = torch.load(rir_path, weights_only=True)
    speech: Tensor1d = torch.load(speech_path, weights_only=True)
    reverb: Tensor1d = RirConvolution.get_reverb(speech, rir)

    file_name: str = name_generator.next()

    directory = directory / file_name[0] / file_name[1]
    directory = csdir.create_directory(directory.absolute())

    tensor_file: Path = directory / f"{file_name}.reverb.pt"
    torch.save({
        "rir": rir,
        "speech": speech,
        "reverb": reverb,
    }, tensor_file)
    return tensor_file


def main():
    random: Random = Random(split_dataset_config.random_seed)

    print("Shuffling...")
    # 文件名为随机值，排序就是打乱
    rirs: list[str] = load_and_sort(split_dataset_config.rir_contents)
    speeches: list[str] = load_and_sort(split_dataset_config.speech_contents)

    print("Saving train lists...")
    csfile.write_all_lines(split_dataset_config.train_list_rir, rirs[:-20000])
    csfile.write_all_lines(split_dataset_config.train_list_speech, speeches[:-20000])

    validation_files: list[str] = []
    test_files: list[str] = []

    reverb_name_generator: StringRandom = StringRandom(random, 16)
    reverb_contents_file: io.TextIOWrapper
    csdir.create_directory(split_dataset_config.reverb_directory)
    with open(split_dataset_config.reverb_directory.joinpath("contents.csv").absolute(),
              "w", newline="") as reverb_contents_file:
        reverb_contents_writer: CsvWriter = csv.writer(reverb_contents_file)
        reverb_contents_writer.writerow(["Reverb", "Rir", "Speech"])

        print("Genrating validation files...")
        rir_path: str
        speech_path: str
        for rir_path, speech_path in zip(rirs[-20000:-10000], 
                                         speeches[-20000:-10000], 
                                         strict=True):
            print(f"    for {rir_path} + {speech_path}...")
            reverb_path: Path = save_reverb(rir_path, 
                                            speech_path, 
                                            reverb_name_generator, 
                                            split_dataset_config.reverb_directory)
            reverb_contents_writer.writerow([str(reverb_path), rir_path, speech_path])
            reverb_contents_file.flush()
            validation_files.append(str(reverb_path))

        print("Genrating test files...")
        for rir_path, speech_path in zip(rirs[-10000:], 
                                         speeches[-10000:], 
                                         strict=True):
            print(f"    for {rir_path} + {speech_path}...")
            reverb_path = save_reverb(rir_path, 
                                      speech_path, 
                                      reverb_name_generator, 
                                      split_dataset_config.reverb_directory)
            reverb_contents_writer.writerow([str(reverb_path), rir_path, speech_path])
            reverb_contents_file.flush()
            test_files.append(str(reverb_path))

    print("Saving validation lists...")
    csfile.write_all_lines(split_dataset_config.validation_list, validation_files)

    print("Saving test lists...")
    csfile.write_all_lines(split_dataset_config.test_list, test_files)

    print("Completed.")


main()
