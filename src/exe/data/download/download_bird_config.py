import pathlib as _pathlib

from exe import common_configurations as _common_configurations


url_pattern: str = \
    "https://zenodo.org/records/4139416/files/fold{:02d}.zip?download=1"


destination: _pathlib.Path = \
    _common_configurations.data_directory / "raw/bird/"


start_index: int = \
    1
