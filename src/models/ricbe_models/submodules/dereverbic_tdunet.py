from statictorch import Tensor3d
import torch

from models.ricbe_models.submodules.dereverbic_bottleneck import DereverbicBottleneck
from models.ricbe_models.submodules.dereverbic_decoder import DereverbicDecoder
from models.ricbe_models.submodules.dereverbic_encoder import DereverbicEncoder


# The Proposed TDUNET Component
class DereverbicTdunet(torch.nn.Module):
    def __init__(self, channels_input: int, replace_lstm_with_encoder_decoder: bool, simple_decoder: bool):
        super().__init__()

        block_count: int = 5
        channel_step: int = 48

        self.encoder = DereverbicEncoder(block_count, channels_input, channel_step, 1)
        bottleneck_channels: int = block_count * channel_step + channels_input

        self.bottleneck = DereverbicBottleneck(bottleneck_channels, 1, replace_lstm_with_encoder_decoder)
        self.decoder = DereverbicDecoder(block_count, bottleneck_channels, channel_step, 1, simple_decoder, True)

    def forward(self, x: Tensor3d):
        features: list[Tensor3d] = self.encoder(x)
        x = self.bottleneck(features[-1])
        x = self.decoder(x, features)
        return x
