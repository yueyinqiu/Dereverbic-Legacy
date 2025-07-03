from torch import Tensor
import torch


class BerpHilbertTransform(torch.nn.Module):
    def __init__(self, N=None, axis=2) -> None:
        super().__init__()
        self.axis = axis
        self.N = N

    def forward(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
        if torch.is_complex(x):
            raise ValueError("x must be real.")

        if (self.axis < 0) or (self.axis > len(x.shape) - 1):
            raise ValueError(
                f"Invalid axis for shape of x, got axis {self.axis} and shape {x.shape}."
            )

        N: int = x.shape[self.axis] if self.N is None else self.N
        if N <= 0:
            raise ValueError("N must be positive.")

        Xf: Tensor = torch.fft.fft(x.double(), n=N, dim=self.axis)
        h: Tensor = torch.zeros(N, dtype=torch.cfloat, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1 : N // 2] = 2
        else:
            h[0] = 1
            h[1 : (N + 1) // 2] = 2

        if x.dim() > 1:
            ind: list = [None] * x.dim()
            ind[self.axis] = slice(None)
            h = h[tuple(ind)]

        x = torch.fft.ifft(Xf * h, dim=self.axis)
        return x
