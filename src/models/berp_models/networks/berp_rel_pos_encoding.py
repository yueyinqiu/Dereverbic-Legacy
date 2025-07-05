import math
import torch


class BerpRelPosEncoding(torch.nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.pe: torch.Tensor | None = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return self.pe

        pe_positive: torch.Tensor = torch.zeros(x.size(1), self.d_model)
        pe_negative: torch.Tensor = torch.zeros(x.size(1), self.d_model)
        position: torch.Tensor = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe: torch.Tensor = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        return self.pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe: torch.Tensor = self.extend_pe(x)
        pos_emb: torch.Tensor = pe[
            :,
            pe.size(1) // 2 - x.size(1) + 1 : pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb
