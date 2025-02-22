import pathlib as _pathlib
import torch as _torch

from exe.data import split_dataset_config
from exe.ricbe import validate_ricbe_config


device: _torch.device = \
    validate_ricbe_config.device


checkpoints_directory: _pathlib.Path = \
    validate_ricbe_config.checkpoints_directory


test_list: _pathlib.Path = \
    split_dataset_config.test_list
