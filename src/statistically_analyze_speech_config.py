import pathlib as _pathlib
import csdir as _csdir

contents_file: _pathlib.Path = \
    _pathlib.Path("./data/speech/contents.csv")

output_directory: _pathlib.Path = \
    _csdir.create_directory("./data/speech/statistic")
