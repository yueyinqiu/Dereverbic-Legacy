
from pathlib import Path
from typing import TypedDict

import csfile
from torch import Tensor
import torch
from statictorch import Tensor1d, Tensor2d, anify

from audio_processors.rir_convolution import RirConvolution
from inputs_and_outputs.data_providers.data_batch import DataBatch


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

        self._next_random: torch.Generator | None = None

    class StateDict(TypedDict):
        random: Tensor

    def state_dict(self) -> StateDict:
        return {
            "random": self._random.get_state()
        }

    def load_state_dict(self, state_dict: StateDict) -> None:
        self._random.set_state(state_dict["random"])
    
    def get(self) -> DataBatch:
        self._next_random = self._random.clone_state()

        rirs: list[Tensor1d] = []
        speeches: list[Tensor1d] = []

        i: Tensor
        for i in torch.randint(len(self._rirs), [self._batch_size], generator=self._next_random):
            rirs.append(torch.load(self._rirs[int(i)], 
                                    weights_only=True, 
                                    map_location=self._device))

        for i in torch.randint(len(self._speeches), [self._batch_size], generator=self._next_random):
            speeches.append(torch.load(self._speeches[int(i)], 
                                        weights_only=True, 
                                        map_location=self._device))

        rirs_batch: Tensor2d = Tensor2d(torch.stack(anify(rirs)))
        speeches_batch: Tensor2d = Tensor2d(torch.stack(anify(speeches)))
        reverb_batch: Tensor2d = RirConvolution.get_reverb(speeches_batch, rirs_batch)

        return DataBatch(rirs_batch, speeches_batch, reverb_batch)

    def next(self):
        assert self._next_random is not None
        self._random = self._next_random
    