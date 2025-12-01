from examples.mha import causal_softmax_attention

from tilelang.profiler import do_bench

import torch
import torch.nn.functional as F
import math
import triton
import pandas as pd
import os

import time
from termcolor import cprint

def bench_softmaxattention(B, H, Sq, S, D, DV, device='cuda', dtype=torch.float16, require_grad=True):
    
    # init input
    query = torch.randn(B, Sq, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=require_grad)
    value = torch.randn(B, S, H, DV, device=device, dtype=dtype, requires_grad=require_grad)
    do = torch.randn(B, Sq, H, DV, device=device, dtype=dtype, requires_grad=False)

    result_dict = {}
    
    # ours
    if Sq < S:
        # decode
        assert require_grad == False
        attention_module = softmax_attention_decode(B, H, Sq, S, D, DV)
    else:
        attention_module = causal_softmax_attention(B, H, S, D, DV, tune=True)
    def ours():
        o = attention_module(query, key, value)
        return o
    ours_fwd_lat = do_bench(ours)
    if require_grad:
        o = attention_module(query, key, value)
        ours_bwd_lat = do_bench(lambda: o.backward(do, retain_graph=True))
    else:
        ours_bwd_lat = None
    
        
    result_dict["MetaAttention"] = (ours_fwd_lat, ours_bwd_lat)
    
    # FlashAttention-2
    fa2_lat = None
    try:
        from flash_attn import flash_attn_func
        def fa2(dim_padded):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func(
                query_padded,
                key_padded,
                value_padded,
                softmax_scale=(
                    1 / D)**0.5,
                causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_fa2 = max(D, DV)
        fa2_fwd_lat = do_bench(lambda: fa2(dim_padded_fa2))
        if require_grad:
            o_ref = fa2(dim_padded_fa2)
            fa2_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        else:
            fa2_bwd_lat = None
        result_dict["FlashAttention-2"] = (fa2_fwd_lat, fa2_bwd_lat)
    except Exception as e:
        print(f"Warning: FlashAttention-2 not available: {e}")

    # aiter
    try:
        import aiter
        from aiter import flash_attn_func
        def aiter_ref(dim_padded):
            if D < dim_padded:
                query_padded = F.pad(query, (0, dim_padded - D), value=0.)
                key_padded = F.pad(key, (0, dim_padded - D), value=0.)
            else:
                query_padded = query
                key_padded = key
            if DV < dim_padded:
                value_padded = F.pad(value, (0, dim_padded - DV), value=0.)
            else:
                value_padded = value
            o_ref = flash_attn_func(
                query_padded,
                key_padded,
                value_padded,
                causal=True,
                softmax_scale=(
                    1 / D)**0.5,
                causal=True)
            if DV < dim_padded:
                o_ref = o_ref[:, :, :, :DV]
            return o_ref
        
        dim_padded_aiter = max(D, DV)
        aiter_fwd_lat = do_bench(lambda: aiter_ref(dim_padded_aiter))
        if require_grad:
            o_ref = aiter_ref(dim_padded_aiter)
            aiter_bwd_lat = do_bench(lambda: o_ref.backward(do, retain_graph=True))
        else:
            aiter_bwd_lat = None
        result_dict["AIter"] = (aiter_fwd_lat, aiter_bwd_lat)
    except Exception as e:
        print(f"Warning: AIter not available: {e}")
        
    return result_dict

if __name__ == "__main__":
    B_list = [1, 8]
    H = 16
    S_list = [2048, 4096, 8192]
    D = 192
    DV = 128

    results = []
    for B in B_list:
        for S in S_list:
            cprint(f"Benchmarking B={B} S={S}", "cyan")
            res = bench_softmaxattention(B, H, S, S, D, DV, require_grad=True)
            for k, v in res.items():
                fwd_lat, bwd_lat = v
                results.append({
                    "Method": k,
                    "B": B,
                    "S": S,
                    "Forward Latency (ms)": fwd_lat,
                    "Backward Latency (ms)": bwd_lat if bwd_lat is not None else None
                })
    
    df = pd.DataFrame(results)
    print(df)
    
    output_path = "softmax_attention_benchmark_results.csv"
    df.to_csv(output_path, index=False)
    cprint(f"Results saved to {output_path}", "green")