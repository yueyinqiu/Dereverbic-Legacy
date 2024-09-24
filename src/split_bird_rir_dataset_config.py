import pathlib as _pathlib

contents_file: _pathlib.Path = \
    _pathlib.Path("./data/rir/contents.csv")

output_directory: _pathlib.Path = \
    _pathlib.Path("./data/")

train_rate: float = \
    0.7

validation_rate: float = \
    0.15

random_seed: int = \
    3049