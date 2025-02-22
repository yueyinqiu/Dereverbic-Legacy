# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

import torch

from models.fins_models.fins_decoder_block import FinsDecoderBlock


class FinsDecoder(torch.nn.Module):
    def __init__(self, num_filters: int, cond_length: int, rir_length: int):
        super().__init__()
        self.rir_length = rir_length

        self.preprocess = torch.nn.Conv1d(1, 512, kernel_size=15, padding=7)

        self.blocks = torch.nn.ModuleList(
            [
                # 134
                FinsDecoderBlock(512, 512, 1, cond_length),
                # 134
                FinsDecoderBlock(512, 512, 1, cond_length),
                # 134
                FinsDecoderBlock(512, 256, 2, cond_length),
                # 268
                FinsDecoderBlock(256, 256, 2, cond_length),
                # 536
                FinsDecoderBlock(256, 256, 2, cond_length),
                # 1072
                FinsDecoderBlock(256, 128, 3, cond_length),
                # 3216
                FinsDecoderBlock(128, 64, 5, cond_length)
                # 16080
            ]
        )

        self.postprocess = torch.nn.Sequential(torch.nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, v: torch.Tensor, condition: torch.Tensor):
        """
        v: [32, 1, 134]

        condition: [32, 144]
        """
        # [32, 512, 134]
        inputs: torch.Tensor = self.preprocess(v)
        outputs: torch.Tensor = inputs
        layer: torch.nn.Module
        for layer in self.blocks:
            # Final: [32, 64, 16080]
            outputs = layer(outputs, condition)
        # [32, 64, 16000]
        outputs = outputs[:, :, :self.rir_length]
        # [32, 11, 16000]
        outputs = self.postprocess(outputs)

        # [32, 1, 16000]
        direct_early: torch.Tensor = outputs[:, 0:1]
        # [32, 10, 16000]
        late: torch.Tensor = outputs[:, 1:]
        # [32, 10, 16000]
        late = self.sigmoid(late)

        return direct_early, late
