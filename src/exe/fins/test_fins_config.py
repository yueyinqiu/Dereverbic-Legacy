import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.fins import validate_fins_config as _validate_fins_config


device: _torch.device = \
    _validate_fins_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_fins_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list


random_seed: str = \
    "D6E4AC17-1CCF-4FE3-AB31-CC1BDDF61122"
