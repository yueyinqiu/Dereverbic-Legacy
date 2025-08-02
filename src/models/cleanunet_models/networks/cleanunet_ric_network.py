from statictorch import Tensor3d
import torch

from models.cleanunet_models.networks.cleanunet_network import CleanunetNetwork
from models.ricbe_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.ricbe_models.submodules.dereverbic_preprocess import DereverbicPreprocess


class CleanUNetRicNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        channels: int = 48
        self.preprocess = DereverbicPreprocess(2, channels)
        self.cleanunet = CleanunetNetwork(channels, channels * 2)
        self.postprocess = DereverbicPostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d, speech: Tensor3d):
        rs: Tensor3d = Tensor3d(torch.cat([reverb, speech], dim=1))
        rs = self.preprocess(rs)
        rir: Tensor3d = self.cleanunet(rs)
        rir = self.postprocess(rir)
        return rir
