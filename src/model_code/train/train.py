"""
torchrun --nproc_per_node 1 train.py 
"""
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time
import wandb
import datetime
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

import lovely_tensors as lt; lt.monkey_patch()
from model_code.models.llama import Llama
from model_code.utils import set_all_seed, print, to_readable_format
from model_code.dataloader import MicroBatchDataLoader
from model_code.parallel.tensor_parallel import apply_tensor_parallel
from model_code.parallel.data_parallel import DataParallelNative

def train_step(model, dataloader, device):
    acc_loss = 0.0

    required_grad_sync = pgm.process_group_manager.dp_world_size > 1

    # 梯度累积
    for i in range(dataloader.grad_acc_steps):
        batch = next(dataloader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # 到达梯度累积的最后一步，再进行梯度的all-reduce
        if required_grad_sync:
            model.require_backward_grad_sync = (i == dataloader.grad_acc_steps - 1)

        outputs = model(input_ids=input_ids)

        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(seq_len*batch_size, -1)
        loss= F.cross_entropy(outputs, target_ids, reduction='mean') / dataloader.grad_acc_steps

        loss.backward()

        acc_loss += loss.item()

    return acc_loss


import model_code.parallel.process_group_manager as pgm
from model_code.parallel.process_group_manager import setup_process_group_manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Llama Model")

    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)

    # 分布式训练参数
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--dp_size", type=int, default=1, help="Data Parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["1f1b", "afab"]) #pp的优化方式
 

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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1) # 梯度累积步数
    parser.add_argument("--max_tokens", type=int, default=1e6)

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
    setup_process_group_manager(dp_size=args.dp_size, pp_size=args.pp_size, tp_size=args.tp_size)

    is_wandb_rank = pgm.process_group_manager.tp_rank == 0 and pgm.process_group_manager.dp_rank ==0 and pgm.process_group_manager.pp_is_last_stage
    set_all_seed(args.seed)

    if is_wandb_rank and args.use_wandb:
        wandb.init(
            project="learn_dist_train",
            name=f"{args.run_name}_{pgm.process_group_manager}",
            config={
                "tensor_parallel_size": pgm.process_group_manager.tp_world_size,
                "pipeline_parallel_size": pgm.process_group_manager.pp_world_size,
                "data_parallel_size": pgm.process_group_manager.dp_world_size,
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
            },
        )
        
    # 设置模型
    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    model = Llama(config=model_config)

    if pgm.process_group_manager.tp_world_size > 1:
        model = apply_tensor_parallel(model)

    model.to(dtype).to(device)

    if pgm.process_group_manager.dp_world_size > 1:
        model = DataParallelNative(model)
    model.train()

    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    dist.barrier()

    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
        seed=args.seed
    )

    tokens_per_step = dataloader.global_batch_size * args.seq_len
    if pgm.process_group_manager.global_rank == 0:
        print("Tokens per step:", to_readable_format(tokens_per_step), is_print_rank=is_wandb_rank)
 
    trained_token, step = 0, 0

    dist.barrier()

    while trained_token<args.max_tokens:
        step_start_time = time.time()
        optimizer.zero_grad()

        loss = train_step(model, dataloader, device)

        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_token += tokens_per_step
        step+=1

        # DDP实现，需要reset梯度缓存
        if hasattr(model, 'reset'):
            model.reset()

        print(f"[rank {pgm.process_group_manager.global_rank}] Step: {step}, Loss: {loss:.4f}, "
            f"Global batch size (with seq_len): {to_readable_format(tokens_per_step)}, "
            f"Tokens/s: {to_readable_format(tokens_per_step / step_duration)}, "
            f"Tokens/s/GPU: {to_readable_format(tokens_per_step / step_duration / world_size)}, "
            f"Tokens: {to_readable_format(trained_token)}{('/' + to_readable_format(args.max_tokens))}, "
            f"Memory usage: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            , is_print_rank=is_wandb_rank
        )   

        if is_wandb_rank and args.use_wandb:
            wandb.log({"loss": loss, "tokens_per_step": tokens_per_step, "tokens_per_second": tokens_per_step / step_duration,\
                "memory_usage": torch.cuda.memory_reserved() / 1e9, "trained_tokens": tokens_per_step})
                  

    if is_wandb_rank and args.use_wandb:
        wandb.finish()
    dist.destroy_process_group()