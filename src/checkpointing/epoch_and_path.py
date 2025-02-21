from pathlib import Path
from typing import NamedTuple


class EpochAndPath(NamedTuple):
    epoch: int
    path: Path