import pathlib as _pathlib

from exe import common_configurations as _common_configurations


base_url: str = \
    "https://ghp.ci/https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"
#    "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"


destination: _pathlib.Path = \
    _common_configurations.data_directory / "raw/ears/"


# This is to continue downloading when it failed halfway.
start_index: int = \
    1
