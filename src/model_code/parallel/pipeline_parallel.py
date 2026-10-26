import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import model_code.parallel.process_group_manager as pgm

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"
def pipeline_communication(operarion, device, dtype, tensor=None, shapes=None):

    global STEP
    global VERBOSE

    if operation == 'recv_forward':
        if pgm.process_group_manager.pp_is_first_stage: return None

        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank

    elif operation == 'send_forward':
        if pgm.process_group_manager.pp_is_last_stage: return
        dest = pgm.process_group_manager.pp_next_rank

    elif operation == 'recv_backward':
        if pgm.process_group_manager.pp_is_last_stage: retun None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank

    elif operation == 'send_backward':
        # Skip if this is the first pipeline stage (nothing to send backward)
        if pgm.process_group_manager.pp_is_first_stage: return
        dest = pgm.process_group_manager.pp_prev_rank

    is_send = operation.startswith('send')
    peer_rank = dest if is_send else src

    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)

    if VERBOSE:
        print(f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
              f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
              f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}", flush=True)

    [req.wait() for req in dist.batch_isend_irecv(op)]
    torch.cuda.synchronize()

    if VERBOSE: STEP+=1

    return tensor if not is_send else None


class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        layer_distribution = self.distribute_layers(config.num_hidden_layers)
        self.embedding = model.embedding if pgm.process_group_manager.pp_is_first_stage else nn.Identity()
        self.decoer_layers = nn.ModuleDict({str(i): model.decoder_layers[i] for i in layer_distribution})
        self.final_norm = model.final_norm if pgm.process_group_manager.pp_is_last_stage else nn.Identity()
        self.final_proj = nodel.final_proj if pgm.process_group_manager.pp_is_last_stage else nn.Identity()

    def distribute_layers(self, num_layers):
        pp_world_size = pgm.process_group_manager.pp_world_size
        layers_per_gpu = [num_layers // pp_world_size + (1 if i < num_layers % pp_world_size else 0) for i in range(pp_world_size)]
        start_layer = sum(layers_per_gpu[:pgm.process_group_manager.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.process_group_manager.pp_rank]))

    def forward(self, input_ids, position_ids, hidden_states):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embedding()
        for layerin self.decoder_layers.values():
            x = layer(x, position_ids = position_ids)
        x = self.final_norm(x)
        return self.final_proj(x)


    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: 
            input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)

        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None


# 经典GPipe调度：all-forward-all-backward
# 所有micro batch的数据都前向计算完成，再进行反向传播
def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    logging_loss: torch.float32 = 0.0 # 存放累积的梯度
    input_tensors, output_tensors = [], []

    dp_world_size = pgm.process_group_manager.dp_world_size
    required_grad_sync = dp_world_size > 1

    # all-forward: recv前一层的tensor，forward后send给后一层
    # micro-batch 等价于 梯度累积的batch切分
    for _ in range(dataloader.grad_acc_steps):
        input_tensor = pipeline_communication(operarion='recv_forward', shapes=tensor_shapes, device=device, dtype=dtype)
        batch = next(data_loader)
        batch["hidden_states"] = input_tensor.to(device) if input_tensor is not None else input_tensor
        output_tensor = model.forward(input_ids = batch["input_ids"].to(device), position_ids=batch["position_ids"].to(device), hidden_states=batch["hidden_states"])
        pipeline_communication(operarion='send_forward', tensor=output_tensor, device=device, dtype=dtype)

        if pgm.process_group_manager.pp_is_last_stage:
            output_tensor = F.cross_entropy(output_tensor.transpose(1,2), batch["target_ids"].to(device), reduction='mean')
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # all-backward：recv后一层的tensor，计算梯度后send给前一层
    for ith_microbatch in range(data_loader.grad_acc_steps):
        if required_grad_sync:
            is_last_iteration = (ith_microbatch == data_loader.grad_acc_steps - 1)
            model.require_backward_grad_sync = is_last_iteration

        output_tensor_grad = pipeline_communication(operarion='recv_backward', shapes=tensor_shapes, device=device, dtype=dtype)
        intut_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
        pipeline_communication(operarion='send_backward', tensor=input_tensor_grad, device=device, dtype=dtype)

    return logging_loss