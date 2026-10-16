import torch
import torch.nn.functional as F
from torch import nn

class SiluAndMul(nn.Module):
    def forward(self, gate: torch.Tensor, up:torch.Tensor) -> torch.Tensor:
        return F.silu(gate)*up


if __name__=="__main__":
    m = SiluAndMul()
    gate=torch.rand(2,10 ,128)
    up=torch.rand(2,10,128)

    output=m(gate, up)
    print(output.shape)
