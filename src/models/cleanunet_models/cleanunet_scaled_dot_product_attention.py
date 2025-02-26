from torch import Tensor
import torch


class CleanunetScaledDotProductAttention(torch.nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        attn: Tensor = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output: Tensor = torch.matmul(attn, v)
        return output, attn
    