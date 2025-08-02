from statictorch import Tensor3d
import torch


class DereverbicDecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, simple_decoder: bool):
        super().__init__()
        self.with_skip = not simple_decoder

        kernel_size: int = 7
        stride: int = 1
        padding: int = ((kernel_size - 1) // 2) * dilation

        self.upsample = torch.nn.Upsample(scale_factor = 4)
        self.conv1 = torch.nn.Conv1d(in_channels, 
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv2 = torch.nn.Conv1d(out_channels, 
                                     out_channels, 
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)
        self.conv3 = torch.nn.Conv1d(out_channels, 
                                     out_channels,
                                     kernel_size=kernel_size, 
                                     dilation=dilation,
                                     stride=stride,
                                     padding=padding)

        self.prelu1 = torch.nn.PReLU(out_channels)
        self.prelu2 = torch.nn.PReLU(out_channels)
        self.prelu3 = torch.nn.PReLU(out_channels)

    def forward(self, x: Tensor3d):
        y1: Tensor3d = self.upsample(x)
        y1 = self.conv1(y1)
        y1 = self.prelu1(y1)
        y2: Tensor3d = self.conv2(y1)
        y2 = self.prelu2(y2)
        y2 = self.prelu3(self.conv3(y2))
        if self.with_skip:
            return Tensor3d(y1 + y2)
        else:
            return y2
