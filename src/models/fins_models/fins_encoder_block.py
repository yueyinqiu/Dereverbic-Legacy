# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

import torch
from statictorch import Tensor3d


class FinsEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm=True):
        super().__init__()
        if use_batchnorm:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                torch.nn.BatchNorm1d(out_channels, track_running_stats=True),
                torch.nn.PReLU(),
            )
            self.skip_conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                torch.nn.BatchNorm1d(out_channels, track_running_stats=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                torch.nn.PReLU(),
            )
            self.skip_conv = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            )

    def forward(self, x: Tensor3d):
        """
        x: [32, C, L] (L = 80000, 40000, 20000, 10000, 5000, 2500, 1250, 625, 313, 157, 79, 40, 20)
        """
        # [32, C', L / 2]
        out: torch.Tensor = self.conv(x)
        # [32, C', L / 2]
        skip_out: torch.Tensor = self.skip_conv(x)
        # [32, C', L / 2]
        skip_out = out + skip_out
        return skip_out