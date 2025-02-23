import torch


class RicbeEncoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 dilation: int):
        super().__init__()
        kernel_size: int = 7
        stride: int = 4
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     dilation=dilation,
                                     padding=((kernel_size - 1) // 2) * dilation)
        self.prelu1 = torch.nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu1(self.conv1(x))
