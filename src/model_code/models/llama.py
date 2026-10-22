import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from model_code.layers.Attention import Attention
from model_code.layers.RoPE import RotaryEmbedding as RoPE
from model_code.layers.RMSNorm import RMSNorm_mix as RMSNorm
from model_code.layers.Activation import SiluAndMul as Activation
from transformers.models.llama import LlamaConfig

class _TideLinear(nn.Module):
    def __init__(self, weightL: torch.Tensor):
        super().__init__()

        self.weight = weightL

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)

class LlamarMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.activation = Activation()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x), self.up_proj(x)))


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_size = self.head_dim * self.num_heads
        self.kv_size = self.head_dim * self.num_kv_heads

        self.qkv_proj = nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x:torch.Tensor, positions: torch.Tensor, rotary_embedding: RoPE) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))

        q,k = rotary_embedding(positions, q, k)
        return self.o_proj(self.attn(q,k,v).flatten(-2))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()

        self.mlp = LlamarMLP(config)
        self.attention= LlamaAttention(config)

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions:torch.Tensor, hidden_state: torch.Tensor,
                    residual:torch.Tensor | None, rotary_emb: RoPE
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if residual is None:
            residual = hidden_state
            hidden_state = self.input_layernorm(hidden_state)
        else:
            hidden_state, residual = self.input_layernorm(hidden_state, residual)

        hidden_state = self.attention(hidden_state, positions, rotary_emb)
        hidden_state, residual = self.output_layernorm(hidden_state, residual)
        return self.mlp(hidden_state), residual

class LlamaModel(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_embed = RoPE(
            config.head_dim,
            config.head_dim,
            config.max_position_embeddings,
            float(config.rope_parameters["rope_theta"])
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states, residual = self.embed_tokens(input_ids), None

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, self.rotary_embed)

        return self.norm(hidden_states, residual)[0]

class LlamaForCausalLm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.model = LlamaModel(config)

        if getattr(config, "tie_word_embeddings" , False):
            self.lm_head = _TideLinear(self.model.embed_tokens.weight)

        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Llama(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # sanity check
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

        # 复用上面已组装好的完整因果语言模型（embed -> decoder layers -> norm -> lm_head）
        self.model = LlamaForCausalLm(config)

    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(-1), device=input_ids.device).unsqueeze(0)

        hidden_states = self.model(input_ids, position_ids)  # [batch_size, seq_length, hidden_size]
        logits = self.model.compute_logits(hidden_states)

        return logits  # [batch_size, seq_length, vocab_size]


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    config=LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )

    # 同一份config构造官方和自实现，方便对比结构
    # ours=Qwen3ForCausalLm(config).eval()
    official = AutoModelForCausalLM.from_config(config).eval()

    # print("=== 自实现 ===")
    # print(ours)
    print("=== HF官方 ===")
    print(official)

    # input_ids = torch.randint(0, config.vocab_size, (1,8))
    # positions= torch.arange(8).unsqueeze(0)
    # with torch.inference_mode():
    #     hidden = ours(input_ids, positions)
    #     logits = ours.compute_logits(hidden[:, -1])

    # ours_params = sum(p.numel() for p in ours.parameters())
    # official_params= sum(p.numel() for p in official.parameters())
    
