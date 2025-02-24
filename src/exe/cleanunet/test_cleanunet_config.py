import pathlib as _pathlib
import torch as _torch

from exe.cleanunet import validate_cleanunet_config as _validate_cleanunet_config
from exe.data.preprocess import split_dataset_config as _split_dataset_config


device: _torch.device = \
    _validate_cleanunet_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_cleanunet_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list


random_seed: str = \
    "2AFC0115-D0E3-4C9F-847B-6E73EB79DACF"
