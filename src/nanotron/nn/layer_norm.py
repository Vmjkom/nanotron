import torch
from torch import nn


class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )


# This is equivalent to LLaMA RMSNorm
# https://github.com/huggingface/transformers/blob/28952248b19db29ca25ccf34a5eec413376494a9/src/transformers/models/llama/modeling_llama.py#L112
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
