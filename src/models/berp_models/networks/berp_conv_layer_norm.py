import torch


class BerpConvLayerNorm(torch.nn.Module):
    def __init__(self, ch_out: int):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(ch_out, eps=1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x