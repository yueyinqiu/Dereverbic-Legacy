from statictorch import Tensor3d
import torch


class RicbePreprocess(torch.nn.Module):
    def __init__(self, 
                 channels_input: int, 
                 channels_output: int):
        super().__init__()
        self.conv = torch.nn.Conv1d(channels_input, 
                                    channels_output, 
                                    kernel_size=5, 
                                    padding=2, 
                                    stride=1)
        self.prelu = torch.nn.PReLU(channels_output)
        
    def forward(self, x: Tensor3d):
        x = self.conv(x)
        x = self.prelu(x)
        return x
