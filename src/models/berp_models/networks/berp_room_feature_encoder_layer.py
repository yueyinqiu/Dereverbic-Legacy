import torch

from models.berp_models.networks.berp_conv_block import BerpConvBlock
from models.berp_models.networks.berp_pos_wise_feed_forward_module import BerpPosWiseFeedForwardModule
from models.berp_models.networks.berp_residual_connection import BerpResidualConnection
from models.berp_models.networks.berp_xpos_multi_headed_attention import BerpXposMultiHeadedAttention


class BerpRoomFeatureEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 512,
        ch_scale: int = 2,
        dropout_prob: float = 0.1,
        half_step_residual: bool = True
    ):
        super().__init__()

        if half_step_residual:
            feedforward_residual_factor: float = 0.5
        else:
            feedforward_residual_factor = 1.0

        self.ffn1 = BerpResidualConnection(
            module=BerpPosWiseFeedForwardModule(
                encoder_dim=embed_dim,
                expansion_factor=ch_scale,
                dropout_prob=dropout_prob,
            ),
            module_factor=feedforward_residual_factor,
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
        self.self_attn = BerpResidualConnection(
            module=BerpXposMultiHeadedAttention(
                num_heads=num_heads,
                embed_dim=embed_dim,
                dropout_prob=dropout_prob,
            ),
        )
        self.conv_module = BerpResidualConnection(
            module=BerpConvBlock(
                ch_in=embed_dim,
                kernel_size=31,
                dropout_prob=dropout_prob,
            ),
        )

        self.ffn2 = BerpResidualConnection(
            module=BerpPosWiseFeedForwardModule(
                encoder_dim=embed_dim,
                expansion_factor=ch_scale,
                dropout_prob=dropout_prob,
            ),
            module_factor=feedforward_residual_factor,
        )

        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.ffn1(x=x)
        x = self.layer_norm(input=x)
        x = self.self_attn(x=x, key=x, value=x, key_padding_mask=None)
        x = self.conv_module(x=x)
        x = self.ffn2(x=x)
        x = self.final_layer_norm(input=x)
        return x
