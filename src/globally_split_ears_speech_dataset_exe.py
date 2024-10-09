import csv
import random
import csfile
import io
import _csv

import globally_split_ears_speech_dataset_config as config

# All the channels of the same original audio will be always put in the same dataset.

print("Loading...")
tensor_files: list[str] = []

csv_file: 'io.TextIOWrapper[io._WrappedBuffer]'
with open(config.contents_file, newline="") as csv_file:
    csv_reader: '_csv._reader' = csv.reader(csv_file)

    row_str: list[str]
    for row_str in csv_reader:
        assert tuple(row_str) == ("Tensor", "Audio", "Original Audio", "Original Channel")
        break
    
    for row_str in csv_reader:
        tensor_files.append(row_str[0])

print("Shuffling...")
rand: random.Random = random.Random(config.random_seed)
rand.shuffle(tensor_files)

train_count: int = int(config.train_ratio * len(tensor_files))

def save_to_file(file_name: str, tensor_files: list[str]):
    tensor_files.sort()
    assert len(tensor_files) > 0
    csfile.write_all_lines(config.output_directory / file_name, tensor_files)

save_to_file("rir_train.txt", tensor_files[:train_count])
save_to_file("rir_test.txt", tensor_files[train_count:])

print("Completed.")
