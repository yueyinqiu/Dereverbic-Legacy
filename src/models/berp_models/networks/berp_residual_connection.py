import torch


class BerpResidualConnection(torch.nn.Module):
    def __init__(
        self, module: torch.nn.Module, module_factor: float = 1.0, input_factor: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.module(x, *args, **kwargs) * self.module_factor) + (
            x * self.input_factor
        )