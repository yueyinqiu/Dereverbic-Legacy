# This model is modified from: 
# https://github.com/kyungyunlee/fins
# Please respect the original license

import torch


class FinsUpsampleNet(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, upsample_factor: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer: torch.nn.ConvTranspose1d = torch.nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
        )
        torch.nn.init.orthogonal_(layer.weight)
        self.layer = torch.nn.utils.spectral_norm(layer)

    def forward(self, inputs: torch.Tensor):
        outputs: torch.Tensor = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs
