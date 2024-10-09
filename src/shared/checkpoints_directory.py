from pathlib import Path
import typing

class CheckpointsDirectory:
    def __init__(self,
                 path: Path, 
                 prefix: str = "epoch", 
                 suffix: str = ".pt") -> None:
        self._path = path.absolute()
        self._prefix = prefix
        self._suffix = suffix

    def get_path(self, epoch: int) -> Path:
        return self._path / f"{self._prefix}{epoch}{self._suffix}"
    
    def get_all(self) -> typing.Iterator[tuple[int, Path]]:
        file: Path
        for file in self._path.iterdir():
            if not file.is_file():
                continue

            file_name: str = file.name
            if not file_name.startswith(self._prefix):
                continue
            if not file_name.endswith(self._suffix):
                continue
            file_name = file_name[len(self._prefix):-len(self._suffix)]
            
            try:
                epoch: int = int(file_name)
            except ValueError:
                continue
            yield (epoch, file)
    
    def get_latest(self) -> tuple[int, Path]:
        return max(self.get_all(), key=lambda x: x[0])
