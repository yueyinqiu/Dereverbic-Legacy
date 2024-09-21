import pathlib

url_pattern: str = "https://zenodo.org/records/4139416/files/fold{:02d}.zip?download=1"

destination: pathlib.Path = pathlib.Path("./data/raw/bird/")

start_index: int = 1