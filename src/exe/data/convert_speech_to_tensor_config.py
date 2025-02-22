import pathlib as _pathlib
import typing as _typing

from exe import common_configurations
from exe.data import download_ears_config


inputs: _typing.Iterable[_pathlib.Path] = \
    download_ears_config.destination.glob("**/*.wav")


output_directory: _pathlib.Path = \
    common_configurations.data_directory / "speech/"


random_seed: str = \
    "17D29B95-E171-4724-9F7E-E130FB3633D6"
