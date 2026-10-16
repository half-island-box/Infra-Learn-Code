import torch

bn_input=torch.rand((2,3,4,4))

def batch_norm(x:torch.Tensor, eps=1e-05)-> torch.Tensor:
    alpha=torch.nn.Parameter(torch.ones(x.shape[1]))
    beta=torch.nn.Parameter(torch.zeros(x.shape[1]))

    mean=torch.mean(x, dim=(0,2,3), keepdim=True)
    var=torch.var(x, dim=(0,2,3), keepdim=True, unbiased=False)

    output=(x-mean)/torch.sqrt(var+eps)
    output=output*alpha.view(1,-1,1,1)+beta.view(1,-1,1,1)
    return output

bn=torch.nn.BatchNorm2d(3)
print("Torch BN output: \n", bn(bn_input))
print("Out Bn output: \n", batch_norm(bn_input) )
