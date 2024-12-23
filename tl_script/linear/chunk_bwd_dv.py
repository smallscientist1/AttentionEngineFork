import torch
from typing import Optional, Tuple
import torch.nn.functional as F

import tvm.tl.language as T
from tvm import tl

from einops import rearrange

def chunk_bwd_kernel_dv(
    batch, headq, headk, head, seqlen, dim, dimv,
    BT, BK, BV, num_stages = 1, thread_num = 128
):
    NT = seqlen // BT
    NK = dim // BK
    NV = dimv // BV
    dtype = "bfloat16"
    accum_dtype = "float"
    num_stages = num_stages
    thread_num = thread_num

    scale = 1.0
    assert(head % headk == 0)
    head_headk_ratio = head // headk
    assert(head % headq == 0)
    head_headq_ratio = head // headq

    @T.prim_func
    def main(
        q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        
        dv: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NV, NT, batch*head, threads=thread_num) as (bx, by, bz):
            b_dv = T.alloc_fragment((BT, BV), accum_dtype)
            b_dv_shared = T.alloc_shared((BT, BV), dtype)
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g1 = T.alloc_fragment((BT), accum_dtype)
            b_g2 = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_g_last = T.alloc_fragment((1), accum_dtype)
            b_A = T.alloc_fragment((BT, BT), accum_dtype)
            b_A_cast = T.alloc_fragment((BT, BT), dtype)
            b_q_shared = T.alloc_shared((BT, BK), dtype)
            b_q_local = T.alloc_fragment((BT, BK), dtype)
            b_k_shared = T.alloc_shared((BT, BK), dtype)
            b_k_local = T.alloc_fragment((BT, BK), dtype)
            b_dh_shared = T.alloc_shared((BK, BV), dtype)
            b_dh_local = T.alloc_fragment((BK, BV), dtype)
            b_do_shared = T.alloc_shared((BT, BV), dtype)
            
            bhead = bz % head
            bb = bz // head
            bheadk = bhead // head_headk_ratio
            bheadq = bhead // head_headq_ratio
            
            T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
            T.copy(b_g_shared, b_g)
            b_g_last[0] = b_g_shared[BT-1]
            
            T.clear(b_dv)
            T.clear(b_A)
            for i_k in T.Pipelined(NK, num_stages=num_stages):
                T.copy(k[bb, bheadk, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_k_shared)
                T.copy(dh[bb, bhead, by*dim+i_k*BK:by*dim+(i_k+1)*BK, bx*BV:(bx+1)*BV], b_dh_shared)
                T.copy(q[bb, bheadq, by*BT:(by+1)*BT, i_k*BK:(i_k+1)*BK], b_q_shared)
                
                T.gemm(b_k_shared, b_dh_shared, b_dv)
                T.gemm(b_k_shared, b_q_shared, b_A, transpose_B=True)
            for i,j in T.Parallel(BT, BV):
                b_dv[i,j] *= T.exp(-b_g[i] + b_g_last[0])
            T.copy(b_g_shared, b_g1)
            T.copy(b_g_shared, b_g2)
            for i,j in T.Parallel(BT, BT):
                b_A[i,j] *= T.exp(-b_g1[i] + b_g2[j]) * scale
            for i,j in T.Parallel(BT,BT):
                b_A[i,j] = T.if_then_else(
                    i <= j, b_A[i,j], 0
                )
            
            T.copy(do[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV], b_do_shared)
            T.copy(b_A,b_A_cast)
            T.gemm(b_A_cast, b_do_shared, b_dv)
            T.copy(b_dv, b_dv_shared)
            T.copy(b_dv_shared, dv[bb, bhead, by*BT:(by+1)*BT, bx*BV:(bx+1)*BV])
   
    return main
             
if __name__=="__main__":
    torch.cuda.manual_seed(0)
    B, H, D, DV = 4, 20, 128, 64
    HQ, HK = 20,20
    TLen = 4096 # 512

    BT, BK, BV = 64, 64, 64
    num_stages = 2
    num_threads = 128
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"
    
    q = torch.randn(B, HQ, TLen, D, dtype=dtype, device=device)
    k = torch.randn(B, HK, TLen, D, dtype=dtype, device=device)
    g = F.sigmoid(torch.randn(B, H, TLen, dtype=accum_dtype, device=device))
    do = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    dh = torch.randn(B, H, TLen//BT*D, DV, dtype=dtype, device=device)
    
    mod = tl.cached(chunk_bwd_kernel_dv, [5,], B, HQ, HK, H, TLen, D, DV, BT, BK, BV, num_stages, num_threads)
    dv = mod(q, k, g, do, dh)
    
    dh_ref = rearrange(dh, "b h (t d) v -> b h t d v", d=D)
    from fla.ops.simple_gla.chunk import chunk_simple_gla_bwd_dv
    dv_ref = chunk_simple_gla_bwd_dv(q,k,g,do,dh_ref,scale=1.0, chunk_size=BT)
    
    from benchmark.bench_utils import print_debug
    print_debug(dv, dv_ref)
    
    from tvm.tl.utils import do_bench
    def run():
        dv = mod(q, k, g, do, dh)
    def run_ref():
        dv_ref = chunk_simple_gla_bwd_dv(q,k,g,do,dh_ref,scale=1.0, chunk_size=BT)
    
    latency = do_bench(run, warmup=10, rep=10)
    latency_ref = do_bench(run_ref, warmup=10, rep=10)
    print(f"latency: {latency}, latency_ref: {latency_ref}")