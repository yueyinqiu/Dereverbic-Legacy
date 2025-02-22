import pathlib as _pathlib
import torch as _torch
import typing as _typing

from exe.data import split_dataset_config
from exe.ricbe import train_ricbe_config


device: _torch.device = \
    train_ricbe_config.device


checkpoints_directory: _pathlib.Path = \
    train_ricbe_config.checkpoints_directory


start_checkpoint: int = \
    1


validation_list: _pathlib.Path = \
    split_dataset_config.validation_list


rank: _typing.Literal["rir_only", "speech_only", "both"] = \
    "rir_only"