from statictorch import Tensor3d
import torch

from models.ricbe_models.submodules.ricbe_encoder_decoder_pair import RicbeEncoderDecoderPair
from models.ricbe_models.submodules.ricbe_postprocess import RicbePostprocess
from models.ricbe_models.submodules.ricbe_preprocess import RicbePreprocess

class RicbeRicNetwork(torch.nn.Module):
    def __init__(self, replace_lstm_with_encoder_decoder: bool):
        super().__init__()

        channels: int = 48
        self.preprocess_for_rir = RicbePreprocess(2, channels)
        self.pair_for_rir = RicbeEncoderDecoderPair(channels, replace_lstm_with_encoder_decoder)
        self.postprocess_for_rir = RicbePostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d, speech: Tensor3d):
        rs: Tensor3d = Tensor3d(torch.cat([reverb, speech], dim=1))
        rs = self.preprocess_for_rir(rs)
        rir: Tensor3d = self.pair_for_rir(rs)
        rir = self.postprocess_for_rir(rir)
        return rir
    