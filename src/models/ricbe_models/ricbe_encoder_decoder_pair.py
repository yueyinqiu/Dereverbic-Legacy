# Inspired by https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py

from statictorch import Tensor3d
import torch

from models.ricbe_models.ricbe_bottleneck import RicbeBottleneck
from models.ricbe_models.ricbe_decoder import RicbeDecoder
from models.ricbe_models.ricbe_encoder import RicbeEncoder


class RicbeEncoderDecoderPair(torch.nn.Module):
    def __init__(self, channels_input: int):
        super().__init__()

        block_count: int = 5
        channel_step: int = 48

        self.encoder = RicbeEncoder(block_count, channels_input, channel_step, 1)
        bottleneck_channels: int = block_count * channel_step + channels_input

        self.bottleneck = RicbeBottleneck(bottleneck_channels, 1)
        self.decoder = RicbeDecoder(block_count, bottleneck_channels, channel_step, 1)

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = self.encoder(x)
        x = self.bottleneck(features[-1])
        x = self.decoder(x, features)
        return x
