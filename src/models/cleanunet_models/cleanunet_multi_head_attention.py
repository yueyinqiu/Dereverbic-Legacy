from torch import Tensor
import torch

from models.cleanunet_models.cleanunet_scaled_dot_product_attention import CleanunetScaledDotProductAttention


class CleanunetMultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = torch.nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = CleanunetScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        sz_b: int
        len_q: int
        len_k: int
        len_v: int
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual: Tensor = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        attn: Tensor
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = q + residual

        q = self.layer_norm(q)

        return q, attn
