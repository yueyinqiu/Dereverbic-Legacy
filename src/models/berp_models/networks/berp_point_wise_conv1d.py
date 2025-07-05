import torch


class BerpPointWiseConv1d(torch.nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.pointwise = torch.nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(x)
    