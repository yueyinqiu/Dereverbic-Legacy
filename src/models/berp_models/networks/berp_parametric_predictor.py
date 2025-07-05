import torch

from models.berp_models.networks.berp_conv_layer_norm import BerpConvLayerNorm


class BerpParametricPredictor(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        num_layers: int = 2,
        num_channels: int = 384,
        kernel_size: int = 3,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        layer: int
        for layer in range(num_layers):
            in_channels: int = in_dim if layer == 0 else num_channels
            if layer != 0 and layer != num_layers - 1:
                self.conv.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels,
                            num_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                        torch.nn.ReLU(),
                        BerpConvLayerNorm(num_channels),
                        torch.nn.Dropout(dropout_prob),
                        torch.nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size * 2 - 1,
                            stride=3,
                        ),
                        torch.nn.ReLU(),
                        BerpConvLayerNorm(num_channels),
                        torch.nn.Dropout(dropout_prob),
                    )
                )
            else:
                self.conv.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels,
                            num_channels,
                            kernel_size,
                            padding=(kernel_size - 1) // 2,
                        ),
                        torch.nn.ReLU(),
                        BerpConvLayerNorm(num_channels),
                        torch.nn.Dropout(dropout_prob),
                    )
                )

        self.linear = torch.nn.Linear(num_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        layer: torch.nn.Module
        for layer in self.conv:
            x = layer(x)

        x = self.linear(x.permute(0, 2, 1))

        return x.squeeze(-1)
