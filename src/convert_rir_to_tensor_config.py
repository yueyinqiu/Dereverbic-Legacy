import pathlib as _pathlib
import typing as _typing
import csdir as _csdir

import download_bird_config as _download_bird_config

inputs: _typing.Iterable[_pathlib.Path] = \
    _download_bird_config.destination.glob("**/*.flac")

output_directory: _pathlib.Path = \
    _csdir.create_directory("./data/rir/")

sample_rate: int = \
    16000

mutichannel_behavior: _typing.Literal["first_only", "as_mono", "as_many"] = \
    "as_many"

slice: int = \
    16000

save_wav: bool = \
    False

random_seed: int = \
    3407
