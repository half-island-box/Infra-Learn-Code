import torch
from torch import nn

class RMSNorm_mix(nn.Module):
    def __init__(self, hidden_size:int, eps:float=1e-6):
        super().__init__()

        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps


    #残差和Norm结合：融合上一层的残差+本层的pre-norm
    def forward(self, x:torch.Tensor, residual:torch.Tensor | None=None)-> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        original_dtype=x.dtype # 保存原本的x精度

        x32 = x.float() # 将x转换为float32精度 提高精度因为涉及到开根号
        if residual is not None: #残差处理
            x32=x32+residual.float()
            residual=x32.to(original_dtype)

        #核心公式
        rms=torch.rsqrt(x32.pow(2).mean(-1, keepdim=True)+self.eps) #开根号的分母
        x_normed=x32*rms #乘上分子
        out32=x_normed*self.weight.float() #乘上参数因子

        #恢复精度
        out=out32.to(original_dtype)

        return (out, residual) if residual is not None else out

class RMSNorm(nn.Module):
    def __init__(self, hidden_size:int, eps:float=1e-6):
        super().__init__()

        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps


    def forward(self, x:torch.Tensor, )-> torch.Tensor:
        ori_dtype=x.dtype

        x32=x.float()

        rms=torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True)+self.eps)
        x_normed=x32*rms
        out32=x_normed*self.weight.float()

        return out32.to(ori_dtype)


if __name__ =="__main__":
    official = nn.RMSNorm(1024, eps=1e-6)
    my=RMSNorm(1024)

    x=torch.randn(8,128, 1024)
    with torch.no_grad():
        official.weight.copy_(my.weight)

    print(official(x)[0,0,:3])
    print(my(x)[0,0,:3])