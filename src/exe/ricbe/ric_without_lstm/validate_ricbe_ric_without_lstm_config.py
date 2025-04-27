import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.ricbe.ric_without_lstm import train_ricbe_ric_without_lstm_config as _train_ricbe_ric_without_lstm_config


device: _torch.device = \
    _train_ricbe_ric_without_lstm_config.device


checkpoints_directory: _pathlib.Path = \
    _train_ricbe_ric_without_lstm_config.checkpoints_directory


start_checkpoint: int = \
    10000


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list
