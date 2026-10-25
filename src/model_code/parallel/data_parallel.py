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
        self.register_backward_hook(self._allreduce_grads) #每个参数跑一次all_reduce

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


class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        self.params = set(params) # 原始参数
        self.params_with_grad_ready = set() # 标记某个参数是否已经计算好梯度
        self.grad_data = grad_data # 梯度buffer

        self.process_group = process_group # 用于梯度同步的进程组
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.handle = None

        self.reset()

    # 一个桶内的所有参数作 all-reduce
    def sync_gradient(self) -> None:
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)


    def reset(self) -> None:
        self.handle = None
        self.params_with_grad_ready.clear()
        self.grad_data.zero_()

    def wait(self) -> None:

        assert self.handle is not None, "you should launch an allreduce operation before waiting for it to finish"
        self.handle.wait()

    # 某个参数的梯度累加完车给，则标记为ready
    # 桶内所有参数都ready，则触发 sync_gradient
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:

        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)

        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:

        self.params = list(params)

        self.buckets = [] # 桶列表
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)

        self.params_to_bucket_location = {} #记录参数和具体桶的映射（start, end, bucket_idx）
        self.bucket_size = bucket_size # 单桶尺寸
        self.bucket_sizes = None

        self.grad_data_list = []
        self.grad_type = grad_type
        self._initialize_buckets()

    def _initialize_buckets(self) -> None:
        """将模型参数分配给各个桶"""
        cur_bucket_size = 0
        cur_bucket_idx = 0

        for param in self.params:
            if not param.requires_grad:
                continue

            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
                continue

            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()

            else:
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                cur_bucket_size += param.numel()

        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [ [] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)

        # 创建桶 存放梯度张量
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device='cuda'))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))

        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            param.main_grad = self._get_view_from_tensor(self.grad_data_list[bucket_id], param.shape, data_start_index, data_end_index)


    def _get_view_from_tensor(self, tensor: torch.Tensor, shape: torch.Size, start: int, end: int) -> torch.Tensor:
        return tensor[start: end].view(shape)

    def reset(self) -> None:
        for bucket in self.buckets:
            bucket.reset()

    def wait(self) -> None:
        for bucket in self.buckets:
            bucket.wait()

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)      


"""
前向 → 反向传播（逆序）：
  某个参数的梯度算完 → AccumulateGrad 触发 param_hook
    → param.grad 累加进 main_grad（桶 buffer）
    → 该桶内所有参数都 ready 了？→ 异步 all_reduce 这个桶
  ...（后续参数的梯度继续算，与前面桶的通信重叠）
反向全部结束 → _post_backward
  → wait() 所有桶
  → main_grad 拷回 p.grad
优化器 step
"""

class DataParallelBucket(nn.Module):
    def __init__(self, module: nn.Module, bucket_cap_mb = 25, grad_type = torch.float32):
        super().__init__()
        self.module = module

        self.requires_backward_grad_sync = True
        grad_size = 2

        bucket_size = bucket_cap_mb * 1024 *1024 // grad_size
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.dp_group, bucket_size, grad_type)
        self.register_backward_hook()

        self._post_backward_callback_set = False

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)

    def get_flops(self, *args, **kwargs):
        return self.module.get_flops(*args, **kwargs)

    def register_backward_hook(self):
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, param: torch.nn.Parameter, bucket_manager: BucketManager):
        def param_hook(*unused):

            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.requires_backward_grad_sync:
                    if not self._post_backward_callback_set:
                        torch.autograd.Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True

                bucket_manager.mark_param_as_ready(param)
        return param_hook

    def _post_backward(self):
        self.bucket_manager.wait()
        self._post_backward_callback_set = False

        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype)

    def reset(self):
        self.bucket_manager.reset()