from .imports import *
from . import rir_convolve_fft


class DataBatch(NamedTuple):
    rir: Tensor
    speech: Tensor
    reverb: Tensor


class TrainDataProvider:
    def __init__(self, 
                 rirs: Path, 
                 speeches: Path, 
                 batch_size: int,
                 device: torch.device,
                 random_seed: int):
        self._rirs = csfile.read_all_lines(rirs)
        self._speeches = csfile.read_all_lines(speeches)
        self._batch_size = batch_size
        self._device = device
        self._random = torch.Generator().manual_seed(random_seed)

    class StateDict(TypedDict):
        random: Tensor

    def state_dict(self) -> StateDict:
        return {
            "random": self._random.get_state()
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self._random.set_state(state_dict["random"])
    
    def next_batch(self) -> DataBatch:
        rirs: list[Tensor] = []
        speeches: list[Tensor] = []

        i: Tensor
        for i in torch.randint(len(self._rirs), [self._batch_size], generator=self._random):
            rirs.append(torch.load(self._rirs[int(i)], 
                                    weights_only=True, 
                                    map_location=self._device))

        for i in torch.randint(len(self._speeches), [self._batch_size], generator=self._random):
            speeches.append(torch.load(self._speeches[int(i)], 
                                        weights_only=True, 
                                        map_location=self._device))

        rirs_batch: Tensor = torch.stack(rirs)
        speeches_batch: Tensor = torch.stack(speeches)
        reverb_batch: Tensor = rir_convolve_fft.get_reverb(speeches_batch,
                                                           rirs_batch)

        return DataBatch(rirs_batch, speeches_batch, reverb_batch)


class ValidationOrTestDataset(Dataset):
    def __init__(self, 
                 data_list: Path, 
                 device: torch.device):
        self._paths = csfile.read_all_lines(data_list)
        self._device = device

    def __len__(self):
        return self._paths.__len__()

    class DatasetItem(TypedDict):
        rir: Tensor
        speech: Tensor
        reverb: Tensor

    def __getitem__(self, i: int) -> DatasetItem:
        return torch.load(self._paths[i], weights_only=True, map_location=self._device)

    def get_data_loader(self, batch_size: int) -> DataLoader:
        def collate(data: list[ValidationOrTestDataset.DatasetItem]) -> DataBatch:
            rirs: list[Tensor] = []
            speeches: list[Tensor] = []
            reverbs: list[Tensor] = []

            item: ValidationOrTestDataset.DatasetItem
            for item in data:
                rirs.append(item["rir"])
                speeches.append(item["speech"])
                reverbs.append(item["reverb"])
            
            return DataBatch(torch.stack(rirs), 
                             torch.stack(speeches), 
                             torch.stack(reverbs))

        return DataLoader(self, batch_size, False, collate_fn=collate)


def _test_train_data_provider():
    import csfile
    import tempfile

    temp_path: str
    with tempfile.TemporaryDirectory() as temp_path:
        temp_directory: Path = Path(temp_path)
        
        paths: list[str] = []
        i: int
        for i in range(0, 100):
            path: str = str((temp_directory / f"{i}.pt").absolute())
            paths.append(path)
            torch.save(torch.tensor([i, i]), path)
        
        csfile.write_all_lines(temp_directory / "rir_contents.txt", paths[:50])
        csfile.write_all_lines(temp_directory / "speech_contents.txt", paths[50:])

        data_provider: TrainDataProvider = TrainDataProvider(
            temp_directory / "rir_contents.txt",
            temp_directory / "speech_contents.txt",
            5,
            torch.device("cpu"),
            1234)
        
        print(data_provider.next_batch())

        state: TrainDataProvider.StateDict = data_provider.state_dict()
        print(data_provider.next_batch().rir.flatten())
        print(data_provider.next_batch().speech.flatten())

        data_provider = TrainDataProvider(
            temp_directory / "rir_contents.txt",
            temp_directory / "speech_contents.txt",
            5,
            torch.device("cpu"),
            1234)
        data_provider.load_state_dict(state)
        print(data_provider.next_batch().rir.flatten())
        print(data_provider.next_batch().speech.flatten())


if __name__ == "__main__":
    _test_train_data_provider()
