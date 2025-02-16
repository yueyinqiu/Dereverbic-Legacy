import pathlib as _pathlib
import torch as _torch
import typing as _typing

import common_configurations as _common_config
import split_dataset_config as _split_dataset_config
import train_ricbe_config as _train_ricbe_config


device: _torch.device = \
    _train_ricbe_config.device


checkpoints_directory: _pathlib.Path = \
    _train_ricbe_config.checkpoints_directory


start_checkpoint: int = \
    1


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list


rank: _typing.Literal["rir_only", "speech_only", "both"] = \
    "rir_only"