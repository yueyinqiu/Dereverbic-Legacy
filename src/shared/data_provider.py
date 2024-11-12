from pathlib import Path as _Path
from typing import Any as _Any
from typing import Iterator as _Iterator
from typing import NamedTuple as _NamedTuple
from typing import Literal as _Literal
import csfile as _csfile
from torch import Tensor as _Tensor
import torch as _torch
import rir_convolve_fft as _rir_convolve_fft
from torch.utils.data.dataset import Dataset as _Dataset
from torch.utils.data.dataloader import DataLoader as _DataLoader


class DataBatch(_NamedTuple):
    rir: _Tensor
    speech: _Tensor
    reverb: _Tensor


class TrainDataProvider:
    def __init__(self, 
                 rirs: _Path, 
                 speeches: _Path, 
                 batch_size: int,
                 device: _torch.device,
                 random_seed: int):
        self._rirs = _csfile.read_all_lines(rirs)
        self._speeches = _csfile.read_all_lines(speeches)
        self._batch_size = batch_size
        self._device = device
        self._random = _torch.Generator().manual_seed(random_seed)

    def state_dict(self) -> dict[_Literal["random"], _Tensor]:
        return {
            "random": self._random.get_state()
        }

    def load_state_dict(self, state_dict: dict[_Literal["random"], _Tensor]) -> None:
        self._random.set_state(state_dict["random"])
    
    def next_batch(self) -> DataBatch:
        rirs: list[_Tensor] = []
        speeches: list[_Tensor] = []

        i: _Tensor
        for i in _torch.randint(len(self._rirs), [self._batch_size], generator=self._random):
            rirs.append(_torch.load(self._rirs[int(i)], 
                                    weights_only=True, 
                                    map_location=self._device))

        for i in _torch.randint(len(self._speeches), [self._batch_size], generator=self._random):
            speeches.append(_torch.load(self._speeches[int(i)], 
                                        weights_only=True, 
                                        map_location=self._device))

        rirs_batch: _Tensor = _torch.stack(rirs)
        speeches_batch: _Tensor = _torch.stack(speeches)
        reverb_batch: _Tensor = _rir_convolve_fft.get_reverb(speeches_batch,
                                                             rirs_batch)

        return DataBatch(rirs_batch, speeches_batch, reverb_batch)


class ValidationOrTestDataset(_Dataset):
    def __init__(self, 
                 data_list: _Path, 
                 device: _torch.device):
        self._paths = _csfile.read_all_lines(data_list)
        self._device = device

    def __len__(self):
        return self._paths.__len__()

    def __getitem__(self, i: int) -> dict[_Literal["rir", "speech", "reverb"], _Tensor]:
        return _torch.load(self._paths[i], weights_only=True, map_location=self._device)

    def get_data_loader(self, batch_size: int) -> _DataLoader:
        def collate(data: list[dict[_Literal["rir", "speech", "reverb"], _Tensor]]):
            rirs: list[_Tensor] = []
            speeches: list[_Tensor] = []
            reverbs: list[_Tensor] = []

            item: dict[_Literal["rir", "speech", "reverb"], _Tensor]
            for item in data:
                rirs.append(item["rir"])
                speeches.append(item["speech"])
                reverbs.append(item["reverb"])
            
            return _torch.stack(rirs)

        return _DataLoader(self, batch_size, False, collate_fn=collate)


def _test_train_data_provider():
    import csfile
    import tempfile

    temp_path: str
    with tempfile.TemporaryDirectory() as temp_path:
        temp_directory: _Path = _Path(temp_path)
        
        paths: list[str] = []
        i: int
        for i in range(0, 100):
            path: str = str((temp_directory / f"{i}.pt").absolute())
            paths.append(path)
            _torch.save(_torch.tensor([i, i]), path)
        
        csfile.write_all_lines(temp_directory / "rir_contents.txt", paths[:50])
        csfile.write_all_lines(temp_directory / "speech_contents.txt", paths[50:])

        data_provider: TrainDataProvider = TrainDataProvider(
            temp_directory / "rir_contents.txt",
            temp_directory / "speech_contents.txt",
            5,
            _torch.device("cpu"),
            1234)
        
        print(data_provider.next_batch())

        state: dict = data_provider.state_dict()
        print(data_provider.next_batch().rir.flatten())
        print(data_provider.next_batch().speech.flatten())

        data_provider = TrainDataProvider(
            temp_directory / "rir_contents.txt",
            temp_directory / "speech_contents.txt",
            5,
            _torch.device("cpu"),
            1234)
        data_provider.load_state_dict(state)
        print(data_provider.next_batch().rir.flatten())
        print(data_provider.next_batch().speech.flatten())


if __name__ == "__main__":
    _test_train_data_provider()
