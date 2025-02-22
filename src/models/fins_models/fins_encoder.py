# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

from statictorch import Tensor3d
import torch

from models.fins_models.fins_encoder_block import FinsEncoderBlock


class FinsEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_list: list[FinsEncoderBlock] = []
        channels: list[int] = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        i: int
        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm: bool = True
            else:
                use_batchnorm = False
            in_channels: int = channels[i]
            out_channels: int = channels[i + 1]
            curr_block: FinsEncoderBlock = FinsEncoderBlock(in_channels, out_channels, use_batchnorm)
            block_list.append(curr_block)

        self.encode = torch.nn.Sequential(*block_list)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(512, 128)

    def forward(self, x: Tensor3d):
        """
        x: [32, 1, 80000]
        """
        # [32, 512, 10]
        out: torch.Tensor = self.encode(x)
        # [32, 512, 1]
        out = self.pooling(out)
        # [32, 512]
        out = out.squeeze(-1)
        # [32, 128]
        out = self.fc(out)
        return out