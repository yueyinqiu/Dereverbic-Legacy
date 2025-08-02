import pathlib as _pathlib
import torch as _torch

from exe.dereverbic.tdunet_dereverb import validate_tdunet_dereverb_config as _validate_ricbe_dereverb_config
from exe.data.preprocess import split_dataset_config as _split_dataset_config


device: _torch.device = \
    _validate_ricbe_dereverb_config.device


checkpoints_directory: _pathlib.Path = \
    _validate_ricbe_dereverb_config.checkpoints_directory


test_list: _pathlib.Path = \
    _split_dataset_config.test_list
