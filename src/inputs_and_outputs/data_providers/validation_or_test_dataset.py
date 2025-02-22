from pathlib import Path
from typing import TypedDict
import csfile
import torch
from torch.utils.data import Dataset, DataLoader
from statictorch import Tensor1d, Tensor2d, anify

from data_providers.data_batch import DataBatch


class ValidationOrTestDataset(Dataset):
    def __init__(self, 
                 data_list: Path, 
                 device: torch.device):
        self._paths = csfile.read_all_lines(data_list)
        self._device = device

    def __len__(self):
        return self._paths.__len__()

    class DatasetItem(TypedDict):
        rir: Tensor1d
        speech: Tensor1d
        reverb: Tensor1d

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
            
            return DataBatch(Tensor2d(torch.stack(anify(rirs))), 
                             Tensor2d(torch.stack(anify(speeches))), 
                             Tensor2d(torch.stack(anify(reverbs))))

        return DataLoader(self, batch_size, False, collate_fn=collate)
