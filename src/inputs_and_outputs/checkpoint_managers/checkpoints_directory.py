from pathlib import Path
from typing import Iterable

import csdir

from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath


class CheckpointsDirectory:
    def __init__(self,
                 path: Path | str, 
                 prefix: str = "epoch_", 
                 suffix: str = ".pt") -> None:
        self._path = csdir.create_directory(path).absolute()
        self._prefix = prefix
        self._suffix = suffix

    def get_path(self, epoch: int | None) -> Path:
        if epoch is None:
            return self._path
        return self._path / f"{self._prefix}{epoch}{self._suffix}"
    
    def _get_all_not_sorted(self) -> Iterable[EpochAndPath]:
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
            yield EpochAndPath(epoch, file)
    
    def get_all(self) -> list[EpochAndPath]:
        return sorted(self._get_all_not_sorted(), key=lambda x: x[0])
    
    def get_latest(self) -> EpochAndPath | None:
        try:
            return max(self._get_all_not_sorted(), key=lambda x: x[0])
        except ValueError:
            return None
