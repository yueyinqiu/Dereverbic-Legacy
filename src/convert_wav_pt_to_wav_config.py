import pathlib as _pathlib
import typing as _typing


# Each element means a file to convert: (input wav pt file, output wav file).
files: _typing.Iterable[tuple[_pathlib.Path,_pathlib.Path]] = \
    [(_pathlib.Path("./data/speech/g/m/gmohgjulpnyhkomn.wav.pt"), 
      _pathlib.Path("./data/speech/g/m/gmohgjulpnyhkomn.wav"))]
