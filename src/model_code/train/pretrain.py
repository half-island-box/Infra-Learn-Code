"""
最小可运行的预训练脚本。

单卡：
    cd src
    python -m model_code.train.pretrain --data_path ../data/sample.jsonl

多卡（DDP，示例 2 卡）：
    cd src
    torchrun --nproc_per_node=2 -m model_code.train.pretrain --data_path ../data/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen3Config

# 兼容直接运行和作为模块运行
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model_code.data.pretrain_dataset import PretrainDataset  # noqa: E402
from model_code.models.qwen3 import Qwen3ForCausalLm  # noqa: E402


# ---------------------------------------------------------------------------
# 工具类 / 函数
# ---------------------------------------------------------------------------


class DummyScaler:
    """当不使用 AMP 时的占位 scaler，保持与 GradScaler 同样的调用接口。"""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass


class ByteTokenizer:
    """
    最小可用的 tokenizer：每个 byte 映射到一个 id。
    vocab_size 必须和模型 config 一致。
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False, max_length: int | None = None, truncation: bool = True):
        encoded = text.encode("utf-8", errors="ignore")
        input_ids = [min(b, self.vocab_size - 1) for b in encoded]
        if max_length is not None and truncation:
            input_ids = input_ids[:max_length]
        if add_special_tokens:
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        return SimpleNamespace(input_ids=input_ids)


def init_model(config: Qwen3Config, from_weight: str | None = None, device: str | torch.device = "cpu"):
    """初始化模型和 tokenizer。"""
    model = Qwen3ForCausalLm(config)
    if from_weight is not None:
        state = torch.load(from_weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    model.to(device)
    tokenizer = ByteTokenizer(config.vocab_size)
    return model, tokenizer


class CausalLMWrapper(nn.Module):
    """
    给 Qwen3ForCausalLm 套上 labels / attention_mask 接口，并返回 loss。
    原模型只输出 hidden_states，这里计算 logits 和交叉熵损失。
    """

    def __init__(self, model: Qwen3ForCausalLm):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        # positions: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden_states)

        loss = None
        if labels is not None:
            # 标准 causal LM 偏移：用 [0..S-2] 预测 [1..S-1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        aux_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        return SimpleNamespace(loss=loss, logits=logits, aux_loss=aux_loss)


def get_lr(step: int, total_steps: int, max_lr: float, min_lr: float, warmup_steps: int) -> float:
    """warmup + cosine decay 学习率。"""
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))

    if total_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ensure_sample_data(data_path: str, num_samples: int = 128):
    """如果数据文件不存在，生成一份极简的随机样本，保证脚本能跑通。"""
    if os.path.exists(data_path):
        return

    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    words = ["hello", "world", "model", "code", "qwen", "torch", "train", "data"]
    with open(data_path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            text = " ".join([words[i % len(words)] for i in torch.randperm(len(words)).tolist()])
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"[warn] data_path 不存在，已自动生成样本数据: {data_path}")


def train_epoch(epoch: int, loader: DataLoader, iters: int, start_step: int = 0):
    start_time = time.time()
    model.train()

    for step, (input_ids, attn_mask, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device, non_blocking=True)
        attn_mask = attn_mask.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate, args.min_lr, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]["lr"]
            print(
                f"epoch={epoch} step={step}/{iters} loss={current_loss:.4f} "
                f"lr={current_lr:.2e} time={spend_time:.2f}s"
            )

    # 如果最后一步没凑齐 accumulation_steps，补一次更新
    if (step) % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal pre-training script for Qwen3-like model")

    # 数据 / IO
    parser.add_argument("--data_path", type=str, default="data/sample.jsonl", help="JSONL 训练数据路径")
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain", help="checkpoint 输出目录")
    parser.add_argument("--from_weight", type=str, default=None, help="加载已有权重的路径")

    # 模型 / 训练
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    # 设备 / 精度
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true", help="启用自动混合精度（仅 CUDA 有效）")

    return parser.parse_args()


def main():
    global args, model, optimizer, scaler, autocast_ctx

    args = parse_args()

    # DDP 初始化
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        args.world_size = dist.get_world_size()
    else:
        args.world_size = 1

    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

    # 超小模型配置（演示用）
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )

    raw_model, tokenizer = init_model(config, args.from_weight, device=args.device)
    model = CausalLMWrapper(raw_model)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if local_rank >= 0 else None)

    ensure_sample_data(args.data_path)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    # AMP / scaler
    use_amp = args.use_amp and args.device.startswith("cuda")
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda") if use_amp else DummyScaler()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.output_dir, exist_ok=True)
    iters = len(train_loader)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_epoch(epoch, train_loader, iters)

        # 保存 checkpoint（保存底层 Qwen3ForCausalLm 的权重，方便后续直接加载）
        if is_main:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            torch.save(raw_model.state_dict(), save_path)
            print(f"saved checkpoint -> {save_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
