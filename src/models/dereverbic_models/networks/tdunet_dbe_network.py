from statictorch import Tensor3d
import torch

from models.dereverbic_models.submodules.dereverbic_tdunet import DereverbicTdunet
from models.dereverbic_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.dereverbic_models.submodules.dereverbic_preprocess import DereverbicPreprocess

class TdunetDbeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48

        self.preprocess = DereverbicPreprocess(1, channels)
        self.pair = DereverbicTdunet(channels, False, False)
        self.postprocess = DereverbicPostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d):
        rev: Tensor3d = self.preprocess(reverb)
        rir: Tensor3d = self.pair(rev)
        rir = self.postprocess(rir)
        return rir
    