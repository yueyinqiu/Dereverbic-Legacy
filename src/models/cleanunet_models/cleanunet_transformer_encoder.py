from torch import Tensor
import torch

from models.cleanunet_models.cleanunet_positional_encoding import CleanunetPositionalEncoding
from models.cleanunet_models.ricbe_encoder_layer import CleanunetEncoderLayer


class CleanunetTransformerEncoder(torch.nn.Module):
    def __init__(
            self, d_word_vec=512, n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1, n_position=624, scale_emb=False):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        if n_position > 0:
            self.position_enc = CleanunetPositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_stack = torch.nn.ModuleList([
            CleanunetEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq: Tensor, src_mask, return_attns=False):
        enc_slf_attn_list: list[Tensor] = []

        enc_output: Tensor = src_seq
        if self.scale_emb:
            enc_output = enc_output * self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        enc_layer: torch.nn.Module
        for enc_layer in self.layer_stack:
            enc_slf_attn: Tensor
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
