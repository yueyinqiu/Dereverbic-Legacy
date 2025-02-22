import pathlib as _pathlib
import torch as _torch

from exe.data import split_dataset_config
from exe.fins import train_fins_config


device: _torch.device = \
    train_fins_config.device


checkpoints_directory: _pathlib.Path = \
    train_fins_config.checkpoints_directory


start_checkpoint: int = \
    1


validation_list: _pathlib.Path = \
    split_dataset_config.validation_list


random_seed: str = \
    "D8BB8437-62E8-493D-AB7F-EA0A161142A0"
