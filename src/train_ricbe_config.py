import pathlib as _pathlib
import torch as _torch

import common_configurations as _common_config
import split_dataset_config as _split_dataset_config


device: _torch.device = \
    _torch.device("cuda", 0) if _torch.cuda.is_available() else _torch.device("cpu")


checkpoint_interval: int = \
    1000


checkpoints_directory: _pathlib.Path = \
    _common_config.checkpoints_directory / "ricbe/"


train_list_rir: _pathlib.Path = \
    _split_dataset_config.train_list_rir


train_list_speech: _pathlib.Path = \
    _split_dataset_config.train_list_speech


random_seed: str = \
    "8D525442-6013-4C86-A12A-001A62B5A799"
