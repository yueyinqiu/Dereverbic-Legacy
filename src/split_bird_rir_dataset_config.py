import pathlib as _pathlib

import shared.common_configurations as _common_config
import convert_rir_to_tensor_config as _convert_rir_to_tensor_config

contents_file: _pathlib.Path = \
    _convert_rir_to_tensor_config.output_directory / "contents.csv"

output_directory: _pathlib.Path = \
    _common_config.data_directory

# train over (train + validation + test)
train_ratio: float = \
    0.7

# validation over (validation + test)
validation_ratio: float = \
    0.5

random_seed: int = \
    3049