# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

import torch

from models.fins_models.fins_conditional_batch_norm_1d import FinsConditionalBatchNorm1d
from models.fins_models.fins_upsample_net import FinsUpsampleNet


class FinsDecoderBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 upsample_factor: int, 
                 condition_length: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = FinsConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            FinsUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = FinsConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = torch.nn.Sequential(
            FinsUpsampleNet(in_channels, in_channels, upsample_factor),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = FinsConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = FinsConditionalBatchNorm1d(out_channels, condition_length)

        self.fourth_stack = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out: torch.Tensor, condition: torch.Tensor):
        inputs: torch.Tensor = enc_out

        outputs: torch.Tensor = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs: torch.Tensor = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs

