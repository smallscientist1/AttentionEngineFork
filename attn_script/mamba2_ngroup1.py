from attn_engine import LinearAttentionEngine
from core.core import SymbolicTensor
from core.core import CustomIO
from core.utils import meta_tensor

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

def eval():
    import itertools
    BHTDDVs = itertools.product(
        [1,8],
        [24, 48, 80],
        [2048,4096,8192],
        [128,],
        [64,]
    )
    HQ, HK = 1, 1
    for B,H,T,D,DV in BHTDDVs:
        qkv_meta = (
            meta_tensor(B, HQ, T, D, dtype=torch.bfloat16),
            meta_tensor(B, HK, T, D, dtype=torch.bfloat16),
            meta_tensor(B, H, T, DV, dtype=torch.bfloat16),
        )
        custom_io = CustomIO(
            {
                "A": (1, "heads"),
                "dt": ("batch", "heads", "seq_len")
            }
        )
        mod = LinearAttentionEngine(qkv_meta,
            decay_mod=decay_mod, v_mod=v_mod,
                                    custom_io = custom_io,
                                    tune=True, tune_filename="mamba2")
        from benchmark.bench_utils import do_bench_mamba
        print(f"B={B}, H={H}, T={T}, D={D}, DV={DV}")
        do_bench_mamba(mod, B, HQ,HK,H, T, D, DV, BT=256)

if __name__ == "__main__":
    B, H, T, D, DV = 8,80, 2048, 128, 64 # bug 16384
    HQ, HK = 1, 1
    dtype = torch.bfloat16
    qkv_meta = (
        meta_tensor(B, HQ, T, D, dtype=dtype),
        meta_tensor(B, HK, T, D, dtype=dtype),
        meta_tensor(B, H, T, DV, dtype=dtype),
    )
    custom_io = CustomIO(
        {
            "A": (1, "heads"),
            "dt": ("batch", "heads", "seq_len")
        }
    )
    mod = LinearAttentionEngine(qkv_meta,
        decay_mod=decay_mod, v_mod=v_mod,
                            custom_io = custom_io,
                            tune=True, tune_filename="mamba2",
                            tune_bwd=True)
    with open("mamba2_tl.py","w") as f:
        f.write(mod.tl_code)

    from benchmark.bench_utils import do_bench_mamba
    do_bench_mamba(mod, B, HQ,HK,H, T, D, DV, BT=256, requires_grad=True)
    # eval()
