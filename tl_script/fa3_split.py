import torch
from benchmark.bench_utils import do_bench_attention_128256
import itertools

if __name__ == "__main__":
    

    BHSDDV = list(
        itertools.product(
            (1,),
            (12,20),
            (2048,4096,32768,65536),
            (128,),
            (256,),
        )
    )
    dtype = torch.bfloat16
    for B,H,S,D,DV in BHSDDV:
        print(f"B={B}, H={H}, S={S}, D={D}, DV={DV}, dtype={dtype}")
        do_bench_attention_128256(B, H, S, D, DV, dtype=dtype)
