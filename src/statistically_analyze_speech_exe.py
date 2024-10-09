import csv
import _csv
import io
import matplotlib.pyplot
import torch
import matplotlib
import csfile

import statistically_analyze_speech_config as config

lengths: list[int] = []

csv_file: 'io.TextIOWrapper[io._WrappedBuffer]'
with open(config.contents_file, newline="") as csv_file:
    csv_reader: '_csv._reader' = csv.reader(csv_file)
    
    row_str: list[str]
    for row_str in csv_reader:
        assert tuple(row_str) == ("Tensor", "Audio", "Original Audio", "Original Channel")
        break
    
    tensor_path: str
    for tensor_path, _, _, _ in csv_reader:
        print(f"Loading {tensor_path} ({csv_reader.line_num}) ...")
        tensor = torch.load(tensor_path, weights_only=True)
        lengths.append(tensor.shape[0])

csfile.write_all_lines(config.output_directory / "lengths.txt", (str(x) for x in lengths))
matplotlib.pyplot.hist(lengths, bins=[0, 0.5e5, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 4.5e5, 5e5])
matplotlib.pyplot.savefig(config.output_directory / "lengths.png")

print("Completed.")
