from pathlib import Path as _Path
from typing import Callable as _Callable
from typing import Any as _Any
import csfile as _csfile
from torch import Tensor as _Tensor
import torch as _torch
from weakref import WeakValueDictionary as _WeakValueDictionary

class WavPtDataProvider:
    def __init__(self, 
                 contents: _Path, 
                 filter: _Callable[[_Tensor, _torch.Generator], _Tensor | None],
                 batch_size: int,
                 device: _torch.device,
                 random_seed: int):
        self._contents = _csfile.read_all_lines(contents)
        self._filter = filter
        self._batch_size = batch_size
        self._device = device
        self._random = _torch.Generator().manual_seed(random_seed)
        self._cache: _WeakValueDictionary[int, _Tensor] = _WeakValueDictionary()

    def state_dict(self) -> dict[str, _Any]:
        return {
            "random": self._random.get_state()
        }

    def load_state_dict(self, state_dict: dict[str, _Any]):
        self._random.set_state(state_dict["random"])
    
    def next_batch(self):
        result: list[_Tensor] = []

        while True:
            rest_count: int = self._batch_size - len(result)
            if rest_count == 0:
                break

            i_tensor: _Tensor
            for i_tensor in _torch.randint(0, len(self._contents), 
                                          [rest_count], 
                                          generator=self._random):
                i: int = int(i_tensor)
                tensor: _Tensor | None = self._cache.get(i)
                if tensor is not None:
                    return tensor

                path: str = self._contents[i]
                tensor = _torch.load(path, weights_only=True)
                assert tensor is not None

                tensor = self._filter(tensor, self._random)
                if tensor is not None:
                    result.append(tensor)
        
        return _torch.stack(result).to(self._device)


def _test():
    import csdir
    import csfile

    directory: _Path = csdir.create_directory("./local_test/non_src/wav_pt_data_provider/")
    
    paths: list[str] = []
    i: int
    for i in range(0, 100):
        path: str = str((directory / f"{i}.pt").absolute())
        paths.append(path)
        _torch.save(_torch.tensor([i, i]), path)
    
    csfile.write_all_lines(directory / "contents.txt", paths)
    data_provider: WavPtDataProvider = WavPtDataProvider(
        directory / "contents.txt",
        lambda x, _: (x * 10) if int(x[0]) < 20 else None,
        5,
        _torch.device("cpu"),
        1234)
    
    print(data_provider.next_batch())

    state: dict = data_provider.state_dict()
    print(data_provider.next_batch().flatten())
    print(data_provider.next_batch().flatten())

    data_provider = WavPtDataProvider(
        directory / "contents.txt",
        lambda x, _: (x * 10) if int(x[0]) < 20 else None,
        5,
        _torch.device("cpu"),
        1234)
    data_provider.load_state_dict(state)
    print(data_provider.next_batch().flatten())
    print(data_provider.next_batch().flatten())


if __name__ == "__main__":
    _test()
