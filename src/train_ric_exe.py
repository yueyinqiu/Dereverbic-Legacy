from shared.checkpoints_directory import CheckpointsDirectory
from pathlib import Path
import csfile


class RirDataset:
    def __init__(self, contents_file: Path, batch_size: int) -> None:
        self._files = csfile.read_all_lines(contents_file)

    def get_next_batch(self):
        self._files


def main():
    import train_ric_config as config

    checkpoints = CheckpointsDirectory(config.checkpoints_directory)


main()
