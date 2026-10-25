import math
from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import model_code.parallel.process_group_manager as pgm

# 将一个tensor的最后一个维度，按照分区数量平均切分
def split_tensor_along_last_dim(tensor, num_partitions):
    last_dim = tensor.dim() - 1
    assert tensor.size()[last_dim] % num_partitions == 0, f"{tensor.size()[last_dim]} is not divisible by {num_partitions}"
    last_dim_size = tensor.size()[last_dim] // num_partitions
    return torch.split(tensor, last_dim_size, dim=last_dim)

# all reduce求和 用于按行切分的合并
class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

# gather拼接 适用于按列切分的合并
class Gather(torch.autograd.Function):
    # 各进程沿张量最后一维拼接起来
    @staticmethod
    def forward(ctx, input:torch.Tensor):
        if pgm.process_group_manager.tp_world_size == 1:
            return input
        last_dim = input.dim()-1

        input = input.contiguous()
        tensor_list = [torch.empty_like(input) for _ in range(pgm.process_group_manager.tp_world_size)]
        tensor_list[pgm.process_group_manager.tp_rank] = input

        dist.all_gather(tensor_list, input, group=pgm.process_group_manager.tp_group)
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output

    # 反向：沿最后一维切分
    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output

        chunks = split_tensor_along_last_dim(grad_output, pgm.process_group_manager.tp_world_size)
        return chunks[pgm.process_group_manager.tp_rank].contiguous()


class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.process_group_manager.tp_world_size == 1:
            return grad_output
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.tp_group)
        return grad_output


# 权重矩阵按列切分：接收维度不变，输出维度均分
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool = False):
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        assert out_features % self.tp_world_size == 0, "Hidden dim must be divisible by the tp world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output

        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # 权重初始化
    # 标准的 Kaiming 均匀初始化：bound = sqrt(1 / in_features)
    def reset_parameters(self):
        if self.tp_world_size == 1:
            k =1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return

        # 先在完整矩阵上初始化参数，再切分
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)

        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    # 计算后拼接 按需拼接分片
    def forward(self, input):
        input_parallel = Copy.apply(input)

        output = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = Gather.apply(output)
        return output

# 按行切分：接收维度切分，各分片的运算结果相加才是最终结果
class RowParallelLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.rank = pgm.process_group_manager.tp_rank

        self.in_features = in_features
        self.out_features = out_features

        assert in_features % self.tp_world_size == 0, "Hidden dim must be divisible by the tp world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))

            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization method used for nn.Linear in PyTorch
        if self.tp_world_size == 1:
            # U(-sqrt(k), sqrt(k))
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            torch.nn.init.uniform_(self.weight, -bound, bound)
            return
    
        # When TP > 1, Initialize master weight
        master_weight = torch.empty(self.out_features, self.in_features, dtype=self.weight.dtype, requires_grad=False)
        # Calculate bound based on master weight's input dimension. U(-sqrt(k), sqrt(k))
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition and take the corresponding partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    # 计算后 必须规约分片结果
    def forward(self, input):
        output_parallel = F.linear(input, self.weight)

        output = Reduce.apply(output_parallel)
        return output if self.bias is None else output + self.bias


# 词表切分: [vocab_size, hidden_size] -> [vocab_size // world_size, hidden_size]
# 每个分片只存一部分词，查不到的置0 最后分片结果相加即为最终
class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        super(VocabParallelEmbedding, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse      

        # 沿词表维度分片
        self.vocab_start_index, self.vocab_end_index = self._vocab_range_from_global_vocab_size(
            self.num_embeddings, pgm.process_group_manager.tp_rank, 
            pgm.process_group_manager.tp_world_size
        )    
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        self.weight = nn.Parameter(torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim))

        self.reset_parameters()

    def _vocab_range_from_global_vocab_size(self, global_vocab_size:int, rank: int, world_size:int):
        assert global_vocab_size % world_size ==0, f"{global_vocab_size} is not divisible by {world_size}"
        per_partition_vocab_size = global_vocab_size // world_size

        start = rank * per_partition_vocab_size
        end = start + per_partition_vocab_size
        return start, end

    def reset_parameters(self):
        if self.tp_world_size == 1:
            # Initialize Vocab embedding with N(0, 1)
            torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)
            return

        # When TP > 1, Initialize master weight
        master_weight = torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.weight.dtype, requires_grad=False)
        torch.nn.init.normal_(master_weight, mean=0.0, std=1.0)
        
        # Split the model into size of self.num_embeddings_per_partition and take the corresponding partition
        weight_list = torch.split(master_weight, self.num_embeddings_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):

        input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)

        masked_input = input.clone() - self.vocab_start_index
        masked_input[input_mask]=0

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse
        )

        output_parallel[input_mask, :] = 0.0
        output = Reduce.apply(output_parallel)
        return output


        

            


def apply_tensor_parallel(model):
    def _replace_module(_module, _linear_proj_name, _style, args={}):
        assert _style in ["column", "row", 'vocab']
        linear_layer = getattr(_module, _linear_proj_name)

        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias = linear_layer.bias is not None,
                gather_output= args.get("gather_output", False) # 默认不聚合，直到 lm_head
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,                
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    # 规定各种层其中的linear的切分方式
    # 可一般认为输入层按列，输出层按行
    # 注意：注意力已拆成 q/k/v 三个独立投影，各自按列切分，GQA 下也能正确对齐本地头数
    module_linear_name_stype_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "o_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    # 模型结构：外层(如 Llama) -> .model(ForCausalLm) -> .model(核心 Model，含 layers/embed_tokens)
    causal_lm = model.model
    transformer = causal_lm.model

    for layer in transformer.layers:
        for module_name, linear_proj_name, style in module_linear_name_stype_mapping_list:
            _replace_module(getattr(layer, module_name), linear_proj_name, style)

    _replace_module(transformer, "embed_tokens", "vocab")

    # 词表未绑定时 lm_head 是独立的 nn.Linear，可列切分后 gather 回完整词表；
    # 绑定时它是 _TideLinear(与 embed_tokens 共享权重)，这里跳过，需要单独处理
    if isinstance(causal_lm.lm_head, nn.Linear):
        _replace_module(causal_lm, "lm_head", "column", args={"gather_output": True})

    return model
