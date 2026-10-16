import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):

    def __init__(self, num_heads:int, head_dim:int, num_kv_heads:int):
        super().__init__()

        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads=num_heads
        self.head_dim=head_dim
        self.num_kv_heads=num_kv_heads

        self.group=num_heads // num_kv_heads
        self.scale=head_dim ** -0.5


    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor) -> torch.Tensor:
        # q: [B,S,H, D_h]
        # kv: [B, S, H_kv, D_h]

        B, S, H, D_h = query.shape

        # transpose: [B, H, S, D] 因为要在 head 维度作矩阵运算
        # reshape: [B, H_kv, G, S, D] 方便和后续同一 KV 组内的 query 复用 K/V
        query=query.transpose(1,2).reshape(B, self.num_kv_heads, self.group, S, D_h)


        # transpose: [B, H_kv, S, D]
        # reshape: [B, H_kv, 1, S, D]
        key=key.transpose(1,2).reshape(B, self.num_kv_heads, 1, S, D_h)
        value=value.transpose(1,2).reshape(B, self.num_kv_heads, 1, S, D_h)


        # 需要掩盖上三角：设为true
        mask= ~torch.ones(S, S, dtype=torch.bool, device=query.device).tril()

        scores=query @ key.transpose(-2, -1) *self.scale

        scores=scores.masked_fill(mask, float("-inf"))

        out=F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype) @ value
        return out.reshape(B, H, S, D_h).transpose(1,2)

if __name__ == "__main__":

    torch.manual_seed(42)

    

    B, S, H, Hkv, D=1, 1024, 16, 8, 128

    # device="cuda"

    

    q=torch.randn(B, S, H, D)

    k=torch.randn(B, S, Hkv, D)

    

    attn=Attention(H, D, Hkv)

    

    out=attn(q,k,k)

    

    reference=F.scaled_dot_product_attention(

    q.transpose(1,2),k.transpose(1,2), k.transpose(1,2),

    is_causal=True, scale=D ** -0.5, enable_gqa=True,

    ).transpose(1,2)

    

    print(torch.allclose(out, reference, atol=1e-4, rtol=1e-3))
