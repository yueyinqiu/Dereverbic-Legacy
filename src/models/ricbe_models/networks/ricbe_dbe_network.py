from statictorch import Tensor3d
import torch

from models.ricbe_models.submodules.ricbe_encoder_decoder_pair import RicbeEncoderDecoderPair
from models.ricbe_models.submodules.ricbe_postprocess import RicbePostprocess
from models.ricbe_models.submodules.ricbe_preprocess import RicbePreprocess

class RicbeDbeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48

        self.preprocess = RicbePreprocess(1, channels)
        self.pair = RicbeEncoderDecoderPair(channels, False)
        self.postprocess = RicbePostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess(reverb)
        rir: Tensor3d = self.pair(rev)
        rir = self.postprocess(rir)
        return rir
    