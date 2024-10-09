import pathlib as _pathlib

import shared.common_configurations as _common_config
import convert_speech_to_tensor_config as _convert_speech_to_tensor_config

contents_file: _pathlib.Path = \
    _convert_speech_to_tensor_config.output_directory / "contents.csv"

output_directory: _pathlib.Path = \
    _common_config.data_directory

# per axies
train_ratio: float = \
    0.7 ** 0.5

validation_ratio: float = \
    0.5

random_seed: int = \
    3050