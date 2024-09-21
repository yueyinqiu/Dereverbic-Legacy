import pathlib

# base_url: str = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p001.zip"
base_url: str = "https://ghp.ci/https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p001.zip"

destination: pathlib.Path = pathlib.Path("./data/raw/ears/")

start_index: int = 1