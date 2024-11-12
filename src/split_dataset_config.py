import pathlib as _pathlib

import shared.common_configurations as _common_config
import convert_rir_to_tensor_config as _convert_rir_to_tensor_config
import convert_speech_to_tensor_config as _convert_speech_to_tensor_config


rir_contents: _pathlib.Path = \
    _convert_rir_to_tensor_config.output_directory / "contents.csv"


speech_contents: _pathlib.Path = \
    _convert_speech_to_tensor_config.output_directory / "contents.csv"


train_list_rir: _pathlib.Path = \
    _common_config.data_directory / "train_rir.txt"


train_list_speech: _pathlib.Path = \
    _common_config.data_directory / "train_speech.txt"


reverb_directory: _pathlib.Path = \
    _common_config.data_directory / "reverb"


validation_list: _pathlib.Path = \
    _common_config.data_directory / "validation.txt"


test_list: _pathlib.Path = \
    _common_config.data_directory / "test.txt"


random_seed: str = \
    "1E94E8A3-DB8A-4576-9453-3ED4A2CBC065"
