import math
import einops
import torch

from models.berp_models.networks.berp_xpos import BerpXpos


class BerpXposMultiHeadedAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        xpos_scale_base: int = 512,
        num_heads: int = 8,
        dropout_prob: float = 0.1,
        self_attention: bool = True,
        subln: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = dropout_prob

        self.self_attention = self_attention

        self.w_k = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_v = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.w_q = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.inner_attn_ln = (
            torch.nn.LayerNorm(self.embed_dim) if subln and self.self_attention else None
        )
        self.dropout_module = torch.nn.Dropout(dropout_prob)
        self.xpos = (
            BerpXpos(self.head_dim, xpos_scale_base) if self.self_attention else None
        )

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_k.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.w_v.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.w_q.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.constant_(self.out_proj.bias, 0.0)

    def attention_ops(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask=None,
        attn_mask=None,
    ):
        q *= self.scaling
        attn_weights: torch.Tensor = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = einops.rearrange(
                attn_weights, "(b h) t s -> b h t s", h=self.num_heads
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                torch.finfo(attn_weights.dtype).min,
            )
            attn_weights = einops.rearrange(attn_weights, "b h t s -> (b h) t s")

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        attn_probs: torch.Tensor = self.dropout_module(attn_weights)
        attn: torch.Tensor = torch.bmm(attn_probs, v)
        attn = einops.rearrange(attn, "(b h) l d -> b l (h d)", h=self.num_heads)

        return attn, attn_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        offset=0,
    ):
        bsz: int
        tgt_len: int
        embed_dim: int
        bsz, tgt_len, embed_dim = query.size()
        src_len: int = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz: int
        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q: torch.Tensor = self.w_q(query)
        k: torch.Tensor = self.w_k(key)
        v: torch.Tensor = self.w_v(value)

        q = einops.rearrange(q, "b l (h d) -> (b h) l d", h=self.num_heads)
        k = einops.rearrange(k, "b l (h d) -> (b h) l d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> (b h) l d", h=self.num_heads)

        if self.xpos is not None:
            k = self.xpos(k, offset=0, downscale=True)
            q = self.xpos(q, offset=offset, downscale=False)

        attn: torch.Tensor
        attn_weights: torch.Tensor
        attn, attn_weights = self.attention_ops(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)

        return attn