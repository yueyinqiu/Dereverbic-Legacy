import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.dereverbic.dereverbic_without_energy_decay import validate_dereverbic_without_energy_decay_config as _validate_ricbe_full_without_energy_decay_config


device: _torch.device = \
    _validate_ricbe_full_without_energy_decay_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_ricbe_full_without_energy_decay_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list
