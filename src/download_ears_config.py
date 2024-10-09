import pathlib as _pathlib
import csdir as _csdir

base_url: str = \
    "https://ghp.ci/https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"
#    "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/"

destination: _pathlib.Path = \
    _csdir.create_directory("./data/raw/ears/")

start_index: int = \
    1