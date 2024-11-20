from attn_engine import LinearAttentionEngine
from core.core import SymbolicTensor

import torch
import torch.nn.functional as F 
import math

"""
Example of mamba2 SSD

fwd:
input:
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, DV]
decay: [B, T, H] or [B, T, H, D] (TODO)
...custom_inputs

output: 
O: [B, H, T, DV]
"""
B, H, T, D, DV = 16, 8, 16384, 128, 64
dtype = torch.bfloat16
dtype_accum = torch.float32
device = "cuda"
q = torch.randn(B, H, T, D, dtype=dtype, device=device)
k = torch.randn(B, H, T, D, dtype=dtype, device=device)
v = torch.randn(B, H, T, DV, dtype=dtype, device=device)

dt_mamba = torch.randn(B, T, H, dtype=dtype, device=device)
factory_kwargs = {"device": device, "dtype": dtype}
dt_min=0.001
dt_max=0.1
dt = torch.exp(
        torch.rand(H, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
dt = torch.clamp(dt, min=1e-4)
dt_bias_mamba = dt + torch.log(-torch.expm1(-dt))
dt_mamba = F.softplus(dt_mamba + dt_bias_mamba)

decay = dt_mamba


# placeholder for custom input tensor
A = SymbolicTensor("A", shape=(1, H))
def decay_mod(decay):
    return (decay*A)# .exp()

mod = LinearAttentionEngine(decay_mod=decay_mod)

A_mamba = torch.randn(H, dtype=dtype, device=device)
o = mod(q, k, v, decay, A=A_mamba)
# do = torch.randn(B, H, T, DV, dtype=dtype, device=device)
# do.backward(o)
