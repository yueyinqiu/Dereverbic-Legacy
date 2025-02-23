from statictorch import Tensor3d
import torch


class RicbePostprocess(torch.nn.Module):
    def __init__(self, 
                 channels_input: int, 
                 channels_output: int, 
                 dilation: int, 
                 stride1: int,
                 stride2: int):
        super().__init__()
        self.kernel_size_1 = 11
        self.kernel_size_2 = 3
        self.conv1 = torch.nn.Conv1d(channels_input, 
                                     channels_input // 4, 
                                     kernel_size=self.kernel_size_1, 
                                     stride=stride1, 
                                     padding=((self.kernel_size_1 - 1) // 2) * dilation)
        self.conv2 = torch.nn.Conv1d(channels_input // 4, 
                                     channels_output, 
                                     kernel_size=self.kernel_size_2, 
                                     stride=stride2, 
                                     padding=((self.kernel_size_2 - 1) // 2) * dilation)
        self.prelu = torch.nn.PReLU(channels_input // 4)
        
    def forward(self, x: Tensor3d):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        return x
