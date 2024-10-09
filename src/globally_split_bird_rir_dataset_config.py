import pathlib as _pathlib

import shared.common_configurations as _common_config
import convert_rir_to_tensor_config as _convert_rir_to_tensor_config

contents_file: _pathlib.Path = \
    _convert_rir_to_tensor_config.output_directory / "contents.csv"

output_directory: _pathlib.Path = \
    _common_config.data_directory

train_ratio: float = \
    0.8

random_seed: str = \
    "1E94E8A3-DB8A-4576-9453-3ED4A2CBC065"
