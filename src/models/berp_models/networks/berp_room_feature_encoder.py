import numpy
import torch

from models.berp_models.networks.berp_room_feature_encoder_layer import BerpRoomFeatureEncoderLayer
from models.berp_models.networks.berp_xpos_multi_headed_attention import BerpXposMultiHeadedAttention


class BerpRoomFeatureEncoder(torch.nn.Module):
    @staticmethod
    def init_bert_params(module):
        def normal_(data):
            # with FSDP, module params will be on CUDA, so we cast them back to CPU
            # so that the RNG is consistent with and without FSDP
            data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

        if isinstance(module, torch.nn.Linear):
            normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Embedding):
            normal_(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, BerpXposMultiHeadedAttention):
            normal_(module.w_q.weight.data)
            normal_(module.w_k.weight.data)
            normal_(module.w_v.weight.data)


    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ch_scale: int = 2,
        dropout: float = 0.1,
        num_layers: int = 8,
        encoder_layerdrop=0.0,
        layer_norm_first=True
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embed_dim
        self.ch_scale = ch_scale
        self.num_heads = num_heads

        self.layers = torch.nn.ModuleList(
            [
                BerpRoomFeatureEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ch_scale=ch_scale,
                    dropout_prob=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm_first = layer_norm_first
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_layerdrop

        self.apply(BerpRoomFeatureEncoder.init_bert_params)

    def forward(self, x):
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        layer: torch.nn.Module
        for _, layer in enumerate(self.layers):
            dropout_probability: float = numpy.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x = layer(x)
        return x
