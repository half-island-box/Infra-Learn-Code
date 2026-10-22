import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）。
    在 Attention 的 Q/K 投影之后、分数计算之前应用。
    """

    def __init__(self, head_size: int, rotary_dim: int, max_position_embedding: int, base: float):
        super().__init__()

        if rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")
        if rotary_dim > head_size:
            raise ValueError("rotary_dim must be <= head_size")

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2

        # θ_i = base^{-2i / rotary_dim}
        inv_freq = base ** (-torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)

        # cache shape: [max_position_embedding, rotary_dim]
        # 前 half 存 cos，后 half 存 sin
        cache = torch.empty(max_position_embedding, rotary_dim, dtype=torch.float32)
        angles = torch.outer(torch.arange(max_position_embedding, dtype=torch.float32), inv_freq)
        cache[:, : self.half] = torch.cos(angles)
        cache[:, self.half :] = torch.sin(angles)

        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = positions.reshape(-1)  # [B*S]
        cos_sin = self.cos_sin_cache.index_select(0, flat).to(query.dtype)

        cos = cos_sin[:, : self.half].unsqueeze(-2)  # [B*S, 1, half]
        sin = cos_sin[:, self.half :].unsqueeze(-2)  # [B*S, 1, half]

        def _apply(x: torch.Tensor) -> torch.Tensor:
            shape = x.shape
            # x: [B, S, H, D] -> [B*S, H, D]
            x = x.reshape(flat.numel(), -1, self.head_size)
            x1 = x[..., : self.half]
            x2 = x[..., self.half : self.rotary_dim]

            # 旋转
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin

            if self.rotary_dim < self.head_size:
                tail = x[..., self.rotary_dim :]
                out = torch.cat([o1, o2, tail], dim=-1)
            else:
                out = torch.cat([o1, o2], dim=-1)

            return out.reshape(shape)

        return _apply(query), _apply(key)
