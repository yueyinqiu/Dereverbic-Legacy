import pathlib as _pathlib
import csdir as _csdir

import shared.common_configurations as _common_config
import split_bird_rir_dataset_config as _split_bird_rir_dataset_config
import split_ears_speech_dataset_config as _split_ears_speech_dataset_config

rir_train_contents: _pathlib.Path = \
    _split_bird_rir_dataset_config.output_directory / "rir_train.txt"

rir_validation_contents: _pathlib.Path = \
    _split_bird_rir_dataset_config.output_directory / "rir_validation.txt"

speech_train_contents: _pathlib.Path = \
    _split_ears_speech_dataset_config.output_directory / "speech_train.txt"

speech_validation_contents: _pathlib.Path = \
    _split_ears_speech_dataset_config.output_directory / "speech_validation.txt"

checkpoints_directory: _pathlib.Path = \
    _common_config.checkpoints_directory / "ric/"
