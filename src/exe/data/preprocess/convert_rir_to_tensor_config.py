import pathlib as _pathlib
import typing as _typing

from exe import common_configurations as _common_configurations
from exe.data.download import download_bird_config as _download_bird_config


inputs: _typing.Iterable[_pathlib.Path] = \
    _download_bird_config.destination.glob("**/*.flac")


output_directory: _pathlib.Path = \
    _common_configurations.data_directory / "rir/"


random_seed: str = \
    "F1F780F3-6EFF-41D0-BA5C-672BB4DDF629"
