from statictorch import Tensor3d
import torch

from models.cleanunet_models.networks.cleanunet_network import CleanunetNetwork
from models.cleanunet_models.networks.cleanunet_ric_network import CleanUNetRicNetwork
from models.ricbe_models.submodules.ricbe_postprocess import RicbePostprocess
from models.ricbe_models.submodules.ricbe_preprocess import RicbePreprocess


class CleanUNetDbeNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        channels: int = 48
        self.preprocess = RicbePreprocess(1, channels)
        self.cleanunet = CleanunetNetwork(channels, channels * 2)
        self.postprocess = RicbePostprocess(channels * 2, 1, 1, 1, 5)

    def forward(self, reverb: Tensor3d):
        reverb = self.preprocess(reverb)
        rir: Tensor3d = self.cleanunet(reverb)
        rir = self.postprocess(rir)
        return rir
