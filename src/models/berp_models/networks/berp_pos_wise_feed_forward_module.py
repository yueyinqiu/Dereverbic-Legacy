import torch

from models.berp_models.networks.berp_swish import BerpSwish

class BerpPosWiseFeedForwardModule(torch.nn.Module):
    def __init__(
        self, encoder_dim: int, expansion_factor: int = 4, dropout_prob: float = 0.1
    ):
        super().__init__()
        self.sequentail = torch.nn.Sequential(
            torch.nn.LayerNorm(encoder_dim),
            torch.nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            BerpSwish(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            torch.nn.Dropout(dropout_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequentail(x)
    