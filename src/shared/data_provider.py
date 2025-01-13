from .imports import *
from .rir_convolve_fft import RirConvolveFft
from .dimension_descriptors import *


class DataBatch(NamedTuple):
    rir: Tensor2d[DBatch, DSample]
    speech: Tensor2d[DBatch, DSample]
    reverb: Tensor2d[DBatch, DSample]


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
        rirs: list[Tensor1d] = []
        speeches: list[Tensor1d] = []

        i: Tensor
        for i in torch.randint(len(self._rirs), [self._batch_size], generator=self._random):
            rirs.append(torch.load(self._rirs[int(i)], 
                                    weights_only=True, 
                                    map_location=self._device))

        for i in torch.randint(len(self._speeches), [self._batch_size], generator=self._random):
            speeches.append(torch.load(self._speeches[int(i)], 
                                        weights_only=True, 
                                        map_location=self._device))

        rirs_batch: Tensor2d[DBatch, DSample] = Tensor2d(
            torch.stack(cast(list[Tensor], rirs)))
        speeches_batch: Tensor2d[DBatch, DSample] = Tensor2d(
            torch.stack(cast(list[Tensor], speeches)))
        reverb_batch: Tensor2d[DBatch, DSample] = RirConvolveFft.get_reverb_batch(speeches_batch,
                                                                               rirs_batch)

        return DataBatch(rirs_batch, speeches_batch, Tensor2d(reverb_batch))


class ValidationOrTestDataset(Dataset):
    def __init__(self, 
                 data_list: Path, 
                 device: torch.device):
        self._paths = csfile.read_all_lines(data_list)
        self._device = device

    def __len__(self):
        return self._paths.__len__()

    class DatasetItem(TypedDict):
        rir: Tensor1d[DSample]
        speech: Tensor1d[DSample]
        reverb: Tensor1d[DSample]

    def __getitem__(self, i: int) -> DatasetItem:
        return torch.load(self._paths[i], weights_only=True, map_location=self._device)

    def get_data_loader(self, batch_size: int) -> DataLoader:
        def collate(data: list[ValidationOrTestDataset.DatasetItem]) -> DataBatch:
            rirs: list[Tensor1d] = []
            speeches: list[Tensor1d] = []
            reverbs: list[Tensor1d] = []

            item: ValidationOrTestDataset.DatasetItem
            for item in data:
                rirs.append(item["rir"])
                speeches.append(item["speech"])
                reverbs.append(item["reverb"])
            
            return DataBatch(Tensor2d(torch.stack(cast(list[Tensor], rirs))), 
                             Tensor2d(torch.stack(cast(list[Tensor], speeches))), 
                             Tensor2d(torch.stack(cast(list[Tensor], reverbs))))

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
