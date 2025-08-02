from statictorch import Tensor3d
import torch

from models.cleanunet_models.networks.cleanunet_network import CleanunetNetwork
from models.cleanunet_models.networks.cleanunet_ric_network import CleanUNetRicNetwork
from models.ricbe_models.networks.tdunet_ric_network import TdunetRicNetwork
from models.ricbe_models.submodules.dereverbic_postprocess import DereverbicPostprocess
from models.ricbe_models.submodules.dereverbic_preprocess import DereverbicPreprocess


class CleanUNetTwoStageNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dereverb = CleanunetNetwork()
        self.ric = CleanUNetRicNetwork()

    def forward(self, reverb: Tensor3d):
        speech: Tensor3d = self.dereverb(reverb)
        rir: Tensor3d = self.ric(reverb, speech)
        return speech, rir
