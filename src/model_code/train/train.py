"""
torchrun --nproc_per_node 1 train.py 
"""
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

from model_code.models.llama import Llama
from utils import set_all_seed, print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Llama Model")

    # 环境参数
    parser.add_argument("--omp_num_threads", type=str, default="1") #OpenMP线程数 1 避免多进程抢CPU
    parser.add_argument("--tokenizers_parallelism", type=str, default="false") #是否打开tokenizer并行

    # 模型参数
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # 训练参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4) # AdamW的学习率
    parser.add_argument("--seq_len", type=int, default=32) #最大序列长度
    parser.add_argument("--micro_batch_size", type=int, default=1) # 单卡单步的batch_size

    # 日志参数
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # 设置 env参数
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    os.environ["DEVICE"]="cuda"

    # 由启动器（torchrun / accelerate）自动注入
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl"
    # 将当前进程的默认CUDA设备设为它负责的那张卡 否则所有进程默认用cuda 0
    torch.cuda.set_device(local_rank) 
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    #分布式训练的“握手”环节：
    # 所有进程在这里汇合，建立通信连接，
    # 之后才能调用 dist.all_reduce、DDP、FSDP 等。
    dist.init_process_group(
        rank=global_rank, 
        world_size=world_size, 
        backend=backend, 
        init_method=f"env://", 
        timeout=datetime.timedelta(minutes=2))


    set_all_seed(args.seed)
    # 设置模型
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    model = Llama(config=model_config)
    model.to(dtype).to(device)
    model.train()

    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()

    input_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)
    target_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)

    optimizer.zero_grad()

    outputs = model(input_ids=input_ids)

    target_ids = target_ids.reshape(-1)
    outputs = outputs.view(-1, model_config.vocab_size)
    loss = F.cross_entropy(outputs, target_ids)

    loss.backward()

    optimizer.step()

    print(f"Loss: {loss.item():.4f}", is_print_rank=(global_rank == 0))
    dist.destroy_process_group()