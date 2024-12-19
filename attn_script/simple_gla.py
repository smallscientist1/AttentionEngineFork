from attn_engine import LinearAttentionEngine
from core.core import SymbolicTensor
from core.core import CustomIO
from core.utils import meta_tensor

import torch
import torch.nn.functional as F 
import math

"""
Example of simple gla

fwd:
input:
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, DV]
decay: [B, H, T]
...custom_inputs

output: 
O: [B, H, T, DV]
"""

D = 128
scale = 1 / D**0.5
def q_mod(q, custom_io):
    return q * scale

if __name__ == "__main__":
    B, H, S, D, DV = 16, 8, 2048, D, 128 # bug 16384
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, D, dtype=dtype),
        meta_tensor(B, H, S, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
        }
    )
    mod = LinearAttentionEngine(qkv_meta, q_mod=q_mod,
                            custom_io = custom_io,
                            tune=True, tune_filename="simple_gla")
    with open("simple_gla_tlcode.py", "w") as f:
        f.write(mod.tl_code)

    from benchmark.bench_utils import do_bench_simple_gla
    do_bench_simple_gla(mod, B, H, S, D, DV, BT=64)
