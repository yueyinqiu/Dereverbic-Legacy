import pathlib as _pathlib

import torch as _torch


data_directory: _pathlib.Path = \
    _pathlib.Path("./data/")


checkpoints_directory: _pathlib.Path = \
    _pathlib.Path("./checkpoints/")


device: _torch.device = \
    _torch.device("cuda", 0) if _torch.cuda.is_available() else _torch.device("cpu")
