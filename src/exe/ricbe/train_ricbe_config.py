import pathlib as _pathlib
import torch as _torch

from exe import common_configurations
from exe.data import split_dataset_config


device: _torch.device = \
    _common_configurations.device


checkpoint_interval: int = \
    1000


checkpoints_directory: _pathlib.Path = \
    common_configurations.checkpoints_directory / "ricbe/"


train_list_rir: _pathlib.Path = \
    split_dataset_config.train_list_rir


train_list_speech: _pathlib.Path = \
    split_dataset_config.train_list_speech


random_seed: str = \
    "8D525442-6013-4C86-A12A-001A62B5A799"
