from statictorch import Tensor3d
import torch

from models.ricbe_models.networks.ricbe_dereverb_network import RicbeDereverbNetwork
from models.ricbe_models.networks.ricbe_ric_network import RicbeRicNetwork

# RICBE Model
class RicbeFullNetwork(torch.nn.Module):
    def __init__(self, dereverb: RicbeDereverbNetwork, ric: RicbeRicNetwork):
        super().__init__()
        self.dereverb = dereverb
        self.ric = ric

    def forward(self, reverb: Tensor3d):
        # DR-IC Framework
        speech: Tensor3d = self.dereverb(reverb)
        rir: Tensor3d = self.ric(reverb, speech)
        return speech, rir
