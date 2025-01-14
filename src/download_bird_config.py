import pathlib as _pathlib

import common_configurations as _common_config


url_pattern: str = \
    "https://zenodo.org/records/4139416/files/fold{:02d}.zip?download=1"


destination: _pathlib.Path = \
    _common_config.data_directory / "raw/bird/"


start_index: int = \
    1
