# This network is based on (with modification of network structure):
# https://github.com/ksasso1028/audio-reverb-removal/blob/main/dereverb/auto_verb.py
# The original project does not explicitly state a license
# Please also respect the original author's rights

from statictorch import Tensor3d
import torch

from models.dereverbic_models.submodules.dereverbic_tdunet import DereverbicTdunet
from models.dereverbic_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.dereverbic_models.submodules.dereverbic_preprocess import DereverbicPreprocess

# The De-Reverberation Module
class TdunetDereverbNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48
        self.preprocess_for_speech = DereverbicPreprocess(1, channels)
        self.pair_for_speech = DereverbicTdunet(channels, False, False)
        self.postprocess_for_speech = DereverbicPostprocess(channels * 2, 1, 1, 1, 1)

    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess_for_speech(reverb)
        spe: Tensor3d = self.pair_for_speech(rev)
        return self.postprocess_for_speech(spe)
    