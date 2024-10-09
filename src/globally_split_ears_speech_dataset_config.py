import pathlib as _pathlib

import shared.common_configurations as _common_config
import convert_speech_to_tensor_config as _convert_speech_to_tensor_config

contents_file: _pathlib.Path = \
    _convert_speech_to_tensor_config.output_directory / "contents.csv"

output_directory: _pathlib.Path = \
    _common_config.data_directory

train_ratio: float = \
    0.8

random_seed: str = \
    "3050B442-4DDC-434A-909D-B3B7981C0ACE"
