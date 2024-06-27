import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class AdaNorm(nn.Module):
    def __init__(self, dim, k: float = 0.1, eps: float = 1e-8, bias: bool = False) -> None:
        super(AdaNorm, self).__init__()
        self.k = k
        self.eps = eps
        self.scale = nn.Parameter(torch.empty(dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.scale)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True) + self.eps
    
        input_norm = (input - mean) * torch.rsqrt(var)
        
        adanorm = self.scale * (1 - self.k * input_norm) * input_norm

        if self.bias is not None:
            adanorm = adanorm + self.bias
    
        return adanorm
