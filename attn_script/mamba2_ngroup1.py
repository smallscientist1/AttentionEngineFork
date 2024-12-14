from attn_engine import LinearAttentionEngine
from core.core import SymbolicTensor
from core.core import CustomIO

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
decay: [B, H, T] or [B, H, T, D] (TODO) float32
...custom_inputs

output: 
O: [B, H, T, DV]
"""

# A = SymbolicTensor("A", shape=(1, H))
def decay_mod(decay, custom_io): # (B,H,seqlen)
    A = custom_io.input_tensors["A"]
    return (decay*A)

# dt = SymbolicTensor("dt", shape=(B, H, T))
def v_mod(v, custom_io): # (B,H,seqlen, D)
    dt = custom_io.input_tensors["dt"]
    return v * dt

if __name__ == "__main__":
    B, H, T, D, DV = 1, 24, 2048, 128, 64 # bug 16384
    HQ, HK = 1, 1
    custom_io = CustomIO(
        {
            "A": (1, H),
            "dt": (B, H, T)
        }
    )
    mod = LinearAttentionEngine(decay_mod=decay_mod, v_mod=v_mod,
                            custom_io = custom_io)
    with open("mamba2_tl.py","w") as f:
        f.write(mod.tl_code)

    from benchmark.bench_utils import do_bench_mamba
    do_bench_mamba(mod, B, HQ,HK,H, T, D, DV, BT=64)
