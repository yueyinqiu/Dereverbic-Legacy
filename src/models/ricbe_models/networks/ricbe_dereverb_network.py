# This network is based on (with modification of network structure):
# https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py
# The original project does not explicitly state a license
# Please also respect the original author's rights

from statictorch import Tensor3d
import torch

from models.ricbe_models.submodules.ricbe_encoder_decoder_pair import RicbeEncoderDecoderPair
from models.ricbe_models.submodules.ricbe_postprocess import RicbePostprocess
from models.ricbe_models.submodules.ricbe_preprocess import RicbePreprocess

class RicbeDereverbNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48
        self.preprocess_for_speech = RicbePreprocess(1, channels)
        self.pair_for_speech = RicbeEncoderDecoderPair(channels)
        self.postprocess_for_speech = RicbePostprocess(channels * 2, 1, 1, 1, 1)

    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess_for_speech(reverb)
        spe: Tensor3d = self.pair_for_speech(rev)
        return self.postprocess_for_speech(spe)
    