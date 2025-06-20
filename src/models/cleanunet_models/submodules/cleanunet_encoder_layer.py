import torch

from models.cleanunet_models.submodules.cleanunet_multi_head_attention import CleanunetMultiHeadAttention
from models.cleanunet_models.submodules.cleanunet_positionwise_feed_forward import CleanunetPositionwiseFeedForward


class CleanunetEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super().__init__()
        self.slf_attn = CleanunetMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = CleanunetPositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output: torch.Tensor
        enc_slf_attn: torch.Tensor
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
