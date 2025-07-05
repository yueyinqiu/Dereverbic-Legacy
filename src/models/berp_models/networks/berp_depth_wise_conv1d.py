import torch


class BerpDepthWiseConv1d(torch.nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        assert (
            ch_out % ch_in == 0
        ), "out_channels should be constant multiple of in_channels"

        self.depthwise = torch.nn.Conv1d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=ch_in,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(x)
    