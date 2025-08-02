# This model is modified from (without modification of its main behavior):
# https://github.com/Alizeded/BERP
# (The only behavior change: Only Th and Tt will be predicted.)
# The original repository is licensed under GPL-3.0
# Please also respect the original author's rights

import torch

from models.berp_models.networks.berp_parametric_predictor import BerpParametricPredictor
from models.berp_models.networks.berp_room_feature_encoder import BerpRoomFeatureEncoder


class BerpJointEstimator(torch.nn.Module):
    def __init__(
        self,
        ch_in: int = 128,
        ch_out: int = 1,
        num_layers: int = 8,
        num_heads: int = 8,
        embed_dim: int = 512,
        ch_scale: int = 4,
        dropout_prob: float = 0.1,
        conv_feature_layers=None,
        num_channels_decoder: int = 384,
        kernel_size_decoder: int = 3,
        num_layers_decoder: int = 3,
        dropout_decoder: float = 0.5,
    ):
        if conv_feature_layers is None:
            conv_feature_layers = [
                (512, 10, 4),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ]
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ch_scale = ch_scale
        self.dropout_prob = dropout_prob

        self.conv_feature_layers = conv_feature_layers

        self.feat_proj = torch.nn.Linear(ch_in, embed_dim)

        # layer normalization
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        # room feature encoder
        self.encoder = BerpRoomFeatureEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ch_scale=ch_scale,
            dropout=dropout_prob,
            num_layers=num_layers,
        )

        self.num_layers_decoder = num_layers_decoder
        self.dropout_decoder = dropout_decoder

        self.num_channels_decoder = num_channels_decoder
        self.kernel_size_decoder = kernel_size_decoder

        self.parametric_predictor_Th = BerpParametricPredictor(
            in_dim=embed_dim,
            out_dim=ch_out,
            num_layers=num_layers_decoder,
            num_channels=num_channels_decoder,
            kernel_size=kernel_size_decoder,
            dropout_prob=dropout_decoder,
        )

        self.parametric_predictor_Tt = BerpParametricPredictor(
            in_dim=embed_dim,
            out_dim=ch_out,
            num_layers=num_layers_decoder,
            num_channels=num_channels_decoder,
            kernel_size=kernel_size_decoder,
            dropout_prob=dropout_decoder,
        )

    def parametric_predictor_forward(self, x: torch.Tensor):
        Th_hat: torch.Tensor = self.parametric_predictor_Th(x)
        Tt_hat: torch.Tensor = self.parametric_predictor_Tt(x)
        return Th_hat, Tt_hat

    def forward(self, source: torch.Tensor):
        x: torch.Tensor = source.permute(0, 2, 1)
        x = self.feat_proj(x)

        x = self.layer_norm(x)
        x = self.encoder(x)

        return self.parametric_predictor_forward(x)
