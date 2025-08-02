from statictorch import Tensor3d
import torch

from models.cleanunet_models.networks.cleanunet_network import CleanunetNetwork
from models.ricbe_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.ricbe_models.submodules.dereverbic_preprocess import DereverbicPreprocess


class CleanUNetDbeNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels: int = 48
        self.preprocess = DereverbicPreprocess(1, channels)
        self.cleanunet = CleanunetNetwork(channels, channels * 2)
        self.postprocess = DereverbicPostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d):
        reverb = self.preprocess(reverb)
        rir: Tensor3d = self.cleanunet(reverb)
        rir = self.postprocess(rir)
        return rir
