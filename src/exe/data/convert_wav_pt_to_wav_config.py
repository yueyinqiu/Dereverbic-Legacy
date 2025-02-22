import pathlib as _pathlib
import typing as _typing


# Each element means a file to convert: (input wav pt file, output wav file).
files: _typing.Iterable[tuple[_pathlib.Path,_pathlib.Path]] = \
    [
        (_pathlib.Path("./data/rir/a/a/aaaamutyxdbqsoxa.wav.pt"), 
         _pathlib.Path("./data/rir/a/a/aaaamutyxdbqsoxa.wav")),
        (_pathlib.Path("./data/speech/a/a/aaazkmywfzfwigic.wav.pt"), 
         _pathlib.Path("./data/speech/a/a/aaazkmywfzfwigic.wav"))
    ]
