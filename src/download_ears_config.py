import pathlib as _pathlib
import csdir as _csdir

base_url: str = \
    "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p001.zip"

destination: _pathlib.Path = \
    _csdir.create_directory("./data/raw/ears/")

start_index: int = \
    1