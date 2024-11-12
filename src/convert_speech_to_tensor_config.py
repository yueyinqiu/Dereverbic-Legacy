import pathlib as _pathlib
import typing as _typing

import shared.common_configurations as _common_config
import download_ears_config as _download_ears_config


inputs: _typing.Iterable[_pathlib.Path] = \
    _download_ears_config.destination.glob("**/*.wav")


output_directory: _pathlib.Path = \
    _common_config.data_directory / "speech/"


random_seed: str = \
    "17D29B95-E171-4724-9F7E-E130FB3633D6"
