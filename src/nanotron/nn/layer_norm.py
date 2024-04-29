import torch
from torch import nn


class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, input):
        hidden_states = input
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TritonLayerNorm(nn.LayerNorm):
    pass
