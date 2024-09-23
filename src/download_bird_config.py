import pathlib as _pathlib
import csdir as _csdir

url_pattern: str = \
    "https://zenodo.org/records/4139416/files/fold{:02d}.zip?download=1"

destination: _pathlib.Path = \
    _csdir.create_directory("./data/raw/bird/")

start_index: int = \
    1
