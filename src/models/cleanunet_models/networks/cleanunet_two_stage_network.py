from statictorch import Tensor3d
import torch

from models.cleanunet_models.networks.cleanunet_network import CleanunetNetwork
from models.cleanunet_models.networks.cleanunet_ric_network import CleanUNetRicNetwork


class CleanUNetTwoStageNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dereverb = CleanunetNetwork()
        self.ric = CleanUNetRicNetwork()

    def forward(self, reverb: Tensor3d):
        speech: Tensor3d = self.dereverb(reverb)
        rir: Tensor3d = self.ric(reverb, speech)
        return speech, rir
