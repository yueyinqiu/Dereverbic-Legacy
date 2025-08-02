from typing import Iterator, Protocol
from statictorch import Tensor3d, anify
import torch

from models.ricbe_models.submodules.dereverbic_encoder_block import DereverbicEncoderBlock


class DereverbicEncoder(torch.nn.Module):
    class EncoderBlockList(Protocol):
        def __iter__(self) -> Iterator[DereverbicEncoderBlock]:
            raise RuntimeError()

    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_increase_per_layer: int, 
                 dilation: int) -> None:
        super().__init__()
        block_list: list[DereverbicEncoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input + channels_increase_per_layer
            block_list.append(DereverbicEncoderBlock(channels_input, channels_next, dilation))
            channels_input = channels_next
        self.blocks: DereverbicEncoder.EncoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = []
        features.append(x)
        block: DereverbicEncoderBlock
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
