import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.ricbe.dereverb import train_ricbe_dereverb_config as _train_ricbe_dereverb_config


device: _torch.device = \
    _train_ricbe_dereverb_config.device


checkpoints_directory: _pathlib.Path = \
    _train_ricbe_dereverb_config.checkpoints_directory


start_checkpoint: int = \
    1


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list
