from statictorch import Tensor3d
import torch

from models.ricbe_models.submodules.dereverbic_tdunet import DereverbicTdunet
from models.ricbe_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.ricbe_models.submodules.dereverbic_preprocess import DereverbicPreprocess

# The RIR Inverse Convolution Module
class TdunetRicNetwork(torch.nn.Module):
    def __init__(self, replace_lstm_with_encoder_decoder: bool, simple_decoder: bool):
        super().__init__()

        channels: int = 48
        self.preprocess_for_rir = DereverbicPreprocess(2, channels)
        self.pair_for_rir = DereverbicTdunet(channels, replace_lstm_with_encoder_decoder, simple_decoder)
        self.postprocess_for_rir = DereverbicPostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d, speech: Tensor3d):
        rs: Tensor3d = Tensor3d(torch.cat([reverb, speech], dim=1))
        rs = self.preprocess_for_rir(rs)
        rir: Tensor3d = self.pair_for_rir(rs)
        rir = self.postprocess_for_rir(rir)
        return rir
    