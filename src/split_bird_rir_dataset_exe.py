import csv
import random
import csfile

import split_bird_rir_dataset_config as config

print("Loading...")
with open(config.contents_file, newline="") as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        assert tuple(row) == ("Tensor", "Audio", "Original Audio", "Original Channel")
        break
    
    class CsvRow:
        def __init__(self, row: list[str]) -> None:
            self.tensor_file: str = row[0]
            self.audio_file: str = row[1]
            self.origianl_audio_file: str = row[2]
            self.origianl_channel_file: str = row[3]

    rows: dict[str, list[CsvRow]] = {}
    for row_str in csv_reader:
        row = CsvRow(row_str)
        if row.origianl_audio_file not in rows:
            rows[row.origianl_audio_file] = []
        rows[row.origianl_audio_file].append(row)

print("Shuffling...")
keys = list(rows.keys())
rand = random.Random(config.random_seed)
rand.shuffle(keys)

train_count = int(config.train_ratio * len(keys))
validation_count = int(config.validation_ratio * (len(keys) - train_count))

def save_to_file(file_name: str, keys: list[str]):
    assert len(keys) > 0
    tensor_file_list: list[str] = []
    for key in keys:
        for row in rows[key]:
            tensor_file_list.append(row.tensor_file)
    rand.shuffle(tensor_file_list)
    csfile.write_all_lines(config.output_directory / file_name, tensor_file_list)

save_to_file("rir_train.txt", keys[:train_count])
save_to_file("rir_validation.txt", keys[train_count:train_count + validation_count])
save_to_file("rir_test.txt", keys[train_count + validation_count:])

print("Completed.")
