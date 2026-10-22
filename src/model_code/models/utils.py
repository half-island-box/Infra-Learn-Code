import torch
import random
import numpy as np
import builtins
import fcntl

def print(*args, is_print_rank=True, **kwargs):
    """ 解决多进程的打印问题"""
    if not is_print_rank: return
    with open(__file__, "r") as fh: # 打开当前脚本文件本身作为锁的载体。
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# seed 要先知道训练中哪些环节依赖随机数：
#参数初始化：模型权重初始值通常是随机的。
#数据顺序：DataLoader 每个 epoch 会 shuffle 数据。
#数据增强：随机裁剪、随机遮挡、随机 mask 等。
#Dropout：训练时按概率随机丢弃神经元。
#采样：比如负采样、对比学习的正负样本构造。
def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def to_readable_format(num, precision=3):
    num_str = str(num)
    length = len(num_str)

    def format_with_precision(main, decimal, suffix):
        if precision ==0:
            return f"{main}{suffix}"
        return f"{main}.{decimal[:precision]}{suffix}"

    if length > 12:  # Trillions
        return format_with_precision(num_str[:-12], num_str[-12:], 'T')
    elif length > 9:  # Billions
        return format_with_precision(num_str[:-9], num_str[-9:], 'B')
    elif length > 6:  # Millions
        return format_with_precision(num_str[:-6], num_str[-6:], 'M')
    elif length > 3:  # Thousands
        return format_with_precision(num_str[:-3], num_str[-3:], 'K')
    else:
        return num_str