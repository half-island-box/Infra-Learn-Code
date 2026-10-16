import torch
import torch.nn as nn

ln_input=torch.rand((2,3,4))

def layer_norm(x: torch.Tensor, eps=1e-05) -> torch.Tensor:
    alpha=nn.Parameter(torch.ones(x.shape[-1]))
    beta=nn.Parameter(torch.zeros(x.shape[-1]))

    mean=torch.mean(x, dim=-1, keepdim=True)
    var=torch.var(x, dim=-1, keepdim=True, unbiased=False)

    output=(x-mean)/torch.sqrt(var+eps)
    output=output*alpha+beta
    return output

ln=nn.LayerNorm(4)
print("torch LN output: \n", ln(ln_input))
print("Our Ln output: \n", layer_norm(ln_input))