from statictorch import Tensor3d
import torch

from models.dereverbic_models.networks.tdunet_dereverb_network import TdunetDereverbNetwork
from models.dereverbic_models.networks.tdunet_ric_network import TdunetRicNetwork

# The Proposed DeReverbIC Model
class DereverbicNetwork(torch.nn.Module):
    def __init__(self, dereverb: TdunetDereverbNetwork, ric: TdunetRicNetwork):
        super().__init__()
        self.dereverb = dereverb
        self.ric = ric

    def forward(self, reverb: Tensor3d):
        # The Proposed Two-Stage Framework
        speech: Tensor3d = self.dereverb(reverb)
        rir: Tensor3d = self.ric(reverb, speech)
        return speech, rir
