# This model is modified from: 
# https://github.com/NVIDIA/CleanUNet
# Please respect the original license

from torch import Tensor
import torch


class CleanunetPositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_in, d_hid)
        self.w_2 = torch.nn.Linear(d_hid, d_in)
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor):
        residual: Tensor = x
        x = self.w_2(torch.nn.functional.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x