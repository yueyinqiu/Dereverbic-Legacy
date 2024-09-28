import csv
import random
import csfile
import pathlib
import io
import _csv

import split_ears_speech_dataset_config as config

# All the channels of the same original audio will be always put in the same dataset.

# The train dataset only contains part of the speakers and also part of the contents,
# while the validation dataset and the test dataset randomly select from the rest.

print("Loading...")
csv_file: io.TextIOWrapper[io._WrappedBuffer]
with open(config.contents_file, newline="") as csv_file:
    csv_reader: _csv._reader = csv.reader(csv_file)
    
    row_str: list[str]
    for row_str in csv_reader:
        assert tuple(row_str) == ("Tensor", "Audio", "Original Audio", "Original Channel")
        break
    
    class CsvRow:
        def __init__(self, row: list[str]) -> None:
            self.tensor_file: str = row[0]
            self.audio_file: str = row[1]
            self.origianl_audio_file: str = row[2]
            self.origianl_channel_file: str = row[3]
        
        def origianl_audio_directory_name(self):
            path: pathlib.Path = pathlib.Path(self.origianl_audio_file)
            return path.parent.name
        
        def origianl_audio_file_name(self):
            path: pathlib.Path = pathlib.Path(self.origianl_audio_file)
            return path.name

    rows: dict[tuple[str, str], list[CsvRow]] = {}
    for row_str in csv_reader:
        row: CsvRow = CsvRow(row_str)
        key: tuple[str, str] = (row.origianl_audio_directory_name(), row.origianl_audio_file_name())
        if key not in rows:
            rows[key] = []
        rows[key].append(row)


print("Shuffling...")
rand: random.Random = random.Random(config.random_seed)

train_keys_1: list[str] = list(set(key for key, _ in rows.keys()))
train_keys_2: list[str] = list(set(key for _, key in rows.keys()))

rand.shuffle(train_keys_1)
rand.shuffle(train_keys_2)
train_keys_1 = train_keys_1[:int(config.train_ratio * len(train_keys_1))]
train_keys_2 = train_keys_2[:int(config.train_ratio * len(train_keys_2))]

train_file_list: list[str] = []
key_1: str
for key_1 in train_keys_1:
    key_2: str
    for key_2 in train_keys_2:
        for row in rows.pop((key_1, key_2)):
            train_file_list.append(row.tensor_file)
assert train_file_list.__len__() > 0
rand.shuffle(train_file_list)
csfile.write_all_lines(config.output_directory / "speech_train.txt", train_file_list)

rest_rows: list[CsvRow] = []
rows_of_a_key: list[CsvRow]
for rows_of_a_key in rows.values():
    for row in rows_of_a_key:
        rest_rows.append(row)

rand.shuffle(rest_rows)
validation_end: int = int(config.validation_ratio * len(rest_rows))
validation_file_list: list[str] = [row.tensor_file for row in rest_rows[:validation_end]]
test_file_list: list[str] = [row.tensor_file for row in rest_rows[validation_end:]]

assert validation_file_list.__len__() > 0
assert test_file_list.__len__() > 0

csfile.write_all_lines(config.output_directory / "speech_validation.txt", validation_file_list)
csfile.write_all_lines(config.output_directory / "speech_test.txt", test_file_list)

print("Completed.")
