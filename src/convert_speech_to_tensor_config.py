import pathlib as _pathlib
import typing as _typing
import csdir as _csdir

import download_ears_config as _download_ears_config
import convert_rir_to_tensor_config as _convert_rir_to_tensor_config

inputs: _typing.Iterable[_pathlib.Path] = \
    _download_ears_config.destination.glob("**/*.wav")

output_directory: _pathlib.Path = \
    _csdir.create_directory("./data/speech/")

sample_rate: int = \
    _convert_rir_to_tensor_config.sample_rate

mutichannel_behavior: _typing.Literal["first_only", "as_mono", "as_many"] = \
    "as_mono"

trim_top_db: float = \
    60

trim_frame_length: int = \
    2048

trim_hop_length: int = \
    512

save_wav: bool = \
    False

random_seed: int = \
    3407
