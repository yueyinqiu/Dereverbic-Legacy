import pathlib as _pathlib
import torch as _torch

import common_configurations as _common_config
import split_dataset_config as _split_dataset_config
import train_fins_config as _train_fins_config


device: _torch.device = \
    _train_fins_config.device


checkpoints_directory: _pathlib.Path = \
    _train_fins_config.checkpoints_directory


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list


random_seed: str = \
    "D8BB8437-62E8-493D-AB7F-EA0A161142A0"
