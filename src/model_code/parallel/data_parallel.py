import contextlib
from typing import List
import torch
import torch.distributed as dist
from torch import nn

import model_code.parallel.process_group_manager as pgm

class DataParallelNative(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # 设置反向传播时是否同步梯度，当用梯度累积时设置为false
        self.require_backward_grad_sync = True
        # 注册反向传播的hook
        self.register_backward_hook(self._allreduce_grads)

    # 每个rank拿不同的数据分片跑相同的前向
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    # 当参数需要梯度更新时，自动调用all-reduce——grads得到梯度平均值
    def register_backward_hook(self, hook):

        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)

    # all-reduce均分梯度
    # 每个rank跑不同的数据分片得到不同的梯度 求平均值即为总数据梯度
    def _allreduce_grads(self, grad):
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.dp_group)
            grad /= pgm.process_group_manager.dp_world_size

        return grad
    