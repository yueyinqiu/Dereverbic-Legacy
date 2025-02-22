# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from typing import Iterator, Protocol
from statictorch import Tensor3d, anify
import torch

from models.ricbe_models.ricbe_encoder_block import RicbeEncoderBlock


class RicbeEncoder(torch.nn.Module):
    class EncoderBlockList(Protocol):
        def __iter__(self) -> Iterator[RicbeEncoderBlock]:
            raise RuntimeError()

    def __init__(self, 
                 block_count: int, 
                 channels_input: int,
                 channels_increase_per_layer: int, 
                 dilation: int) -> None:
        super().__init__()
        block_list: list[RicbeEncoderBlock] = []
        for _ in range(block_count):
            channels_next: int = channels_input + channels_increase_per_layer
            block_list.append(RicbeEncoderBlock(channels_input, channels_next, dilation))
            channels_input = channels_next
        self.blocks: RicbeEncoder.EncoderBlockList = anify(torch.nn.ModuleList(block_list))

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = []
        features.append(x)
        block: RicbeEncoderBlock
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
