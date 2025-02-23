import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config
from exe.fins import validate_fins_config


device: _torch.device = \
    validate_fins_config.device


checkpoints_directory: _pathlib.Path = \
    validate_fins_config.checkpoints_directory


test_list: _pathlib.Path = \
    split_dataset_config.test_list


random_seed: str = \
    "D6E4AC17-1CCF-4FE3-AB31-CC1BDDF61122"
