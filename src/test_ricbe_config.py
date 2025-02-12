import pathlib as _pathlib
import torch as _torch
import typing as _typing

import common_configurations as _common_config
import split_dataset_config as _split_dataset_config
import validate_ricbe_config as _validate_ricbe_config


device: _torch.device = \
    _validate_ricbe_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_ricbe_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list
