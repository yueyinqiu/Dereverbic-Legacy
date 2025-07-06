import pathlib as _pathlib
import typing as _typing
import torch as _torch

from exe.data.download import download_bird_config as _download_bird_config
from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.berp import validate_berp_config as _validate_berp_config


device: _torch.device = \
    _validate_berp_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_berp_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list


rir_map: _pathlib.Path = \
    _split_dataset_config.rir_contents


reverb_map: _pathlib.Path = \
    _split_dataset_config.reverb_directory / "contents.csv"


rir_information: _typing.Iterable[_pathlib.Path] = \
    _download_bird_config.destination.glob("**/*.csv")
