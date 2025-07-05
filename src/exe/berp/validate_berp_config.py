import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.berp import train_berp_config as _train_berp_config


device: _torch.device = \
    _train_berp_config.device


checkpoints_directory: _pathlib.Path = \
    _train_berp_config.checkpoints_directory


start_checkpoint: int = \
    0


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list
