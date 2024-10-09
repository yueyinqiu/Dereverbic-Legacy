import pathlib as _pathlib
import csdir as _csdir

import shared.common_configurations as _common_config

base_url: str = \
    "https://ghp.ci/https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"
#    "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"

destination: _pathlib.Path = \
    _common_config.data_directory / "raw/ears/"

start_index: int = \
    1