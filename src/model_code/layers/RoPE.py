import torch
import torch.nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_size:int, rotary_dim:int, max_position_embedding:int, base:float):
        super().__init__()

        if rotary_dim%2!=0:
            raise ValueError("Rotary dim must be even")
        if rotary_dim > head_size:
            raise ValueError("Rotary dim must be smaller than head size")

        self.head_size=head_size
        self.rotary_dim=rotary_dim
        self.half=rotary_dim//2

        inv_freq = base**(-torch.arange(0, rotary_dim, 2, dtype=torch.float32)/rotary_dim)

        cache=torch.empty(max_position_embedding, rotary_dim, dtype=torch.float32)
        angle=cache[:, :self.half]
        torch.outer(torch.arange(max_position_embedding, dtype=torch.float32), inv_freq, out=angle)
        cache[:, self.half:].copy_(angle)

        # 原地修改cos、sin
        torch.cos_(angle)
        torch.sin_(cache[:, self.half])

        # 注册到buffer
        self.register_buffer("cos_sin_cache", cache, persistent=False)


    def forward(
        self, positions:torch.Tensor, query:torch.Tensor, key:torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor]:

        flat=positions.reshape(-1) #[B,S,D] -> [B*S, D]
        cos_sin=self.cos_sin_cache.index_select(0, flat).to(query.dtype)

        cos=cos_sin[:, :self.half].unsqueeze(-2)
        sin=cos_sin[:, :self.half].unsqueeze(-2)

        def _apply(x:torch.Tensor) -> torch.Tensor:
            shape=x.shape
            x=x.reshape(flat.numl(), -1, self.head_size)
            out=torch.empty_like(x)
            x1, x2= x[...,:self.half], x[..., self.half:self.rotary_dim]
            o1, o2=out[...,:self.half], out[..., self.half:self.rotary_dim]

            tmp=torch.empty_like(o1)
            torch.mul(x1, cos, out=o1)
            torch.mul(x2, sin, out=tmp)
            o1.sub_(tmp)

            torch.mul(x2, cos, out=o2)
            torch.mul(x1, sin, out=tmp)
            o2.add_(tmp)

            if self.rotary_dim < self.head_size:
                out[..., self.rotary_dim:].copy_(x[..., self.rotary_dim:])

            return out.reshape(shape)

        return _apply(query), _apply(key)

    