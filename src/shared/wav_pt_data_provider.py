from pathlib import Path
from typing import Callable, Any
import csfile
from torch import Tensor
import torch
from weakref import WeakValueDictionary

class WavPtDataProvider:
    def __init__(self, 
                 contents: Path, 
                 filter: Callable[[Tensor], Tensor | None],
                 batch_size: int,
                 random_seed: int):
        self._contents = csfile.read_all_lines(contents)
        self._filter = filter
        self._batch_size = batch_size
        self._random = torch.Generator().manual_seed(random_seed)
        self._cache: WeakValueDictionary[int, torch.Tensor] = WeakValueDictionary()

    def state_dict(self) -> dict[str, Any]:
        return {
            "contents": self._contents,
            "random": self._random.get_state()
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._contents = state_dict["contents"]
        self._random.set_state(state_dict["random"])
    
    def next_batch(self):
        result: list[Tensor] = []

        while True:
            rest_count: int = self._batch_size - len(result)
            if rest_count == 0:
                break

            i_tensor: torch.Tensor
            for i_tensor in torch.randint(0, len(self._contents), 
                                          [rest_count], 
                                          generator=self._random):
                i: int = int(i_tensor)
                tensor: Tensor | None = self._cache.get(i)
                if tensor is not None:
                    return tensor

                path: str = self._contents[i]
                tensor = torch.load(path, weights_only=True)
                assert tensor is not None

                tensor = self._filter(tensor)
                if tensor is not None:
                    result.append(tensor)
        
        return torch.stack(result)


def _test():
    import csdir
    import csfile

    directory: Path = csdir.create_directory("./local_test/non_src/wav_pt_data_provider/")
    
    paths: list[str] = []
    i: int
    for i in range(0, 100):
        path: str = str((directory / f"{i}.pt").absolute())
        paths.append(path)
        torch.save(torch.tensor([i, i]), path)
    
    csfile.write_all_lines(directory / "contents.txt", paths)
    data_provider: WavPtDataProvider = WavPtDataProvider(
        directory / "contents.txt",
        lambda x: (x * 10) if int(x[0]) < 20 else None,
        5,
        1234)
    
    print(data_provider.next_batch())

    state: dict = data_provider.state_dict()
    print(data_provider.next_batch().flatten())
    print(data_provider.next_batch().flatten())

    data_provider = WavPtDataProvider(
        directory / "contents.txt",
        lambda x: (x * 10) if int(x[0]) < 20 else None,
        5,
        1234)
    data_provider.load_state_dict(state)
    print(data_provider.next_batch().flatten())
    print(data_provider.next_batch().flatten())


if __name__ == "__main__":
    _test()
