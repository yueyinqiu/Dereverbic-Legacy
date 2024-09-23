import pathlib as _pathlib
import typing as _typing

import convert_rir_to_tensor_config as _convert_rir_to_tensor_config

files: _typing.Iterable[tuple[_pathlib.Path,_pathlib.Path]] = \
    [(_pathlib.Path("./data/speech/g/m/gmohgjulpnyhkomn.wav.pt"), 
      _pathlib.Path("./data/speech/g/m/gmohgjulpnyhkomn.wav"))]

sample_rate: int = \
  _convert_rir_to_tensor_config.sample_rate
