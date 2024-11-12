import pathlib as _pathlib
import torch as _torch

import shared.common_configurations as _common_config
import convert_rir_to_tensor_config as _convert_rir_to_tensor_config
import split_dataset_config as _split_bird_rir_dataset_config
import split_speech_dataset_config as _split_ears_speech_dataset_config

device: _torch.device = \
    _torch.device("cuda")

checkpoint_interval: int = \
    5
#    200

validation_interval: int = \
    2
#    100

checkpoints_directory: _pathlib.Path = \
    _common_config.checkpoints_directory / "fins/"

rir_train_contents: _pathlib.Path = \
    _split_bird_rir_dataset_config.output_directory / "rir_train.txt"

rir_validation_contents: _pathlib.Path = \
    _split_bird_rir_dataset_config.output_directory / "rir_validation.txt"

rir_length: int = \
    _convert_rir_to_tensor_config.slice

speech_train_contents: _pathlib.Path = \
    _split_ears_speech_dataset_config.output_directory / "speech_train.txt"

speech_validation_contents: _pathlib.Path = \
    _split_ears_speech_dataset_config.output_directory / "speech_validation.txt"

speech_length: int = \
    16000 * 10

batch_size: int = \
    16

epoch_count: int = \
    -1

random_seed: str = \
    "AB866237-508D-4D3B-AC74-5CB8A84E632B"
