from typing import Iterator, Protocol
from statictorch import Tensor3d, anify
import torch

from models.ricbe_models.submodules.ricbe_decoder_block import RicbeDecoderBlock



class RicbeDecoder(torch.nn.Module):
    class DecoderBlockList(Protocol):
        def __iter__(self) -> Iterator[RicbeDecoderBlock]:
            raise RuntimeError()
        
    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_decrease_per_layer: int, 
                 dilation: int,
                 simple_decoder: bool,
                 concatenate_last: bool = True):
        super().__init__()
        self.concatenate_last: bool = concatenate_last
        block_list: list[RicbeDecoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input - channels_decrease_per_layer
            block_list.append(RicbeDecoderBlock(channels_input * 2, 
                                                channels_next, 
                                                dilation, 
                                                simple_decoder))
            channels_input = channels_next
        self.blocks: RicbeDecoder.DecoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d, features: list[Tensor3d]):
        i: int = -1
        block: RicbeDecoderBlock
        for block in self.blocks:
            x = Tensor3d(torch.cat([x, features[i]], dim=1))
            x = Tensor3d(block(x))
            i = i - 1
            x = Tensor3d(x[:, :, :features[i].size(2)])
        if self.concatenate_last:
            return Tensor3d(torch.cat([x, features[i]], dim=1))
        else:
            return Tensor3d(x)
