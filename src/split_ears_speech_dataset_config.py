import pathlib as _pathlib

contents_file: _pathlib.Path = \
    _pathlib.Path("./data/speech/contents.csv")

output_directory: _pathlib.Path = \
    _pathlib.Path("./data/")

# per axies
train_ratio: float = \
    0.7 ** 0.5

validation_ratio: float = \
    0.5

random_seed: int = \
    3050