import torch

from models.berp_models.networks.berp_depth_wise_conv1d import BerpDepthWiseConv1d
from models.berp_models.networks.berp_point_wise_conv1d import BerpPointWiseConv1d
from models.berp_models.networks.berp_swish import BerpSwish
from models.berp_models.networks.berp_transpose import BerpTranspose


class BerpConvBlock(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        kernel_size: int = 31,
        dropout_prob: float = 0.1,
        expansion_factor: int = 2,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be odd number"
        assert expansion_factor == 2, "Currently expansion factor is 2"

        self.conv = torch.nn.Sequential(
            torch.nn.LayerNorm(ch_in),
            BerpTranspose(shape=(1, 2)),
            BerpPointWiseConv1d(ch_in, ch_in * expansion_factor, stride=1),
            torch.nn.GLU(dim=1),
            BerpDepthWiseConv1d(
                ch_in, ch_in, kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
            torch.nn.BatchNorm1d(ch_in),
            BerpSwish(),
            BerpPointWiseConv1d(ch_in, ch_in, stride=1),
            torch.nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x.transpose(1, 2)