import pathlib as _pathlib
import torch as _torch

import shared.common_configurations as _common_config
import split_dataset_config as _split_dataset_config


device: _torch.device = \
    _torch.device("cuda", 0) if _torch.cuda.is_available() else _torch.device("cpu")


checkpoint_interval: int = \
    1000


checkpoints_directory: _pathlib.Path = \
    _common_config.checkpoints_directory / "fins_modified_reverb/"


train_list_rir: _pathlib.Path = \
    _split_dataset_config.train_list_rir


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list


train_list_speech: _pathlib.Path = \
    _split_dataset_config.train_list_speech


random_seed: str = \
    "AB866237-508D-4D3B-AC74-5CB8A84E632B"
