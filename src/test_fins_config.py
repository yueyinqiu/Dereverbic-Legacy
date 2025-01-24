import pathlib as _pathlib
import torch as _torch
import typing as _typing

import common_configurations as _common_config
import split_dataset_config as _split_dataset_config
import validate_fins_config as _validate_fins_config


device: _torch.device = \
    _validate_fins_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_fins_config.checkpoints_directory


checkpoints: _typing.Iterable[int] = \
    [10000]


test_list: _pathlib.Path = \
    _split_dataset_config.test_list


random_seed: str = \
    "D6E4AC17-1CCF-4FE3-AB31-CC1BDDF61122"
