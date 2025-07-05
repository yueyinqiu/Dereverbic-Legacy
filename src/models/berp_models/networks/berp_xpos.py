import torch


class BerpXpos(torch.nn.Module):
    @staticmethod
    def duplicate_interleave(m: torch.Tensor):
        dim0: int = m.shape[0]
        m = m.view(-1, 1)  # flatten the matrix
        m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
        m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
        return m
    
    @staticmethod
    def rotate_every_two(x: torch.Tensor):
        x1: torch.Tensor = x[:, :, ::2]
        x2: torch.Tensor = x[:, :, 1::2]
        x = torch.stack((-x2, x1), dim=-1)
        return x.flatten(-2)
    
    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, scale: torch.Tensor):
        sin, cos = map(lambda t: BerpXpos.duplicate_interleave(t * scale), (sin, cos))
        return (x * cos) + (BerpXpos.rotate_every_two(x) * sin)

    @staticmethod
    def fixed_pos_embedding(x: torch.Tensor):
        seq_len: int
        dim: int
        seq_len, dim = x.shape
        inv_freq: torch.Tensor = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
        sinusoid_inp: torch.Tensor = torch.einsum(
            "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
        ).to(x)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base

        self.scale: torch.Tensor
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x: torch.Tensor, offset=0, downscale=False):
        length: int = x.shape[1]
        min_pos: int = -(length + offset) // 2
        max_pos: int = length + offset + min_pos
        scale: torch.Tensor = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin: torch.Tensor
        cos: torch.Tensor
        sin, cos = BerpXpos.fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = BerpXpos.apply_rotary_pos_emb(x, sin, cos, scale)
        return x