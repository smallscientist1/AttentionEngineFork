import torch
from typing import Optional, Tuple

import tvm.tl.language as T
from tvm import tl

from einops import rearrange

def chunk_bwd_dh_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    states_in_fp32: bool = False,
    offsets: Optional[torch.Tensor] = None,
    c_offsets: Optional[torch.Tensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HQ = q.shape[2]
    BT = chunk_size
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, c_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        if c_offsets is None:
            c_offsets = torch.cat([offsets.new_tensor([0]), triton.cdiv(offsets[1:] - offsets[:-1], BT)]).cumsum(-1)
        NT = c_offsets[-1]
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    # number of groups in GQA
    NG = HQ // H

    if head_first:
        dh = k.new_empty(B, HQ, NT, K, V, dtype=k.dtype if not states_in_fp32 else torch.float32)
    else:
        dh = k.new_empty(B, NT, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float32)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None

    grid = (NK, NV, N * H)
    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        offsets=offsets,
        c_offsets=c_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NG=NG,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        HEAD_FIRST=head_first,
    )
    return dh, dh0


def chunk_bwd_kernel_dh(
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

    @T.prim_func
    def main(
        q: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        k: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        v: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        
        dh: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
    ):
        with T.Kernel(NK, NV, batch*head, threads=thread_num) as (bx, by, bz):
            b_dh = T.alloc_fragment((BK, BV), accum_dtype)
            dh_shared = T.alloc_shared((BK, BV), dtype)
            q_shared = T.alloc_shared((BT, BK), dtype)
            q_local = T.alloc_fragment((BT, BK), dtype)
            q_local_T = T.alloc_fragment((BK, BT), dtype)
            do_shared = T.alloc_shared((BT, BV), dtype)

            bg_local = T.alloc_fragment((BT), accum_dtype)
            bg_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            bg_last = T.alloc_fragment((1), accum_dtype)

            bhead = bz % head
            bb = bz // head

            T.clear(b_dh)
            loop_st = 0
            loop_ed = NT
            for i_t0 in T.Pipelined(NT, num_stages=num_stages):
                i_t = loop_ed - 1 - i_t0

                # T.copy(b_dh, dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV]) # implicit cast
                T.copy(b_dh, dh_shared)
                T.copy(dh_shared,  dh[bb, bhead, (i_t*dim + bx*BK):(i_t*dim + (bx+1)*BK), by*BV:(by+1)*BV])

                T.copy(q[bb, bhead, i_t*BT:(i_t+1)*BT, bx*BK:(bx+1)*BK], q_shared)
                T.copy(do[bb, bhead, i_t*BT:(i_t+1)*BT, by*BV:(by+1)*BV], do_shared)
                
                T.copy(q_shared, q_local)

                T.copy(g[bb, bhead, i_t*BT:(i_t+1)*BT], bg_shared)

                # q_mod 
                for i,j in T.Parallel(BT, BK):
                    q_local[i,j] *= scale

                T.copy(bg_shared, bg_local)
                bg_last[0] = bg_shared[BT-1]
                for i,j in T.Parallel(BK, BT):
                    q_local_T[i,j] = q_local[j,i] * T.exp(bg_local[j])

                for i,j in T.Parallel(BK, BV):
                    b_dh[i,j] *= T.exp(bg_last[0])
                
                T.gemm(q_local_T, do_shared, b_dh, transpose_B=False)
        
    return main

if __name__ == "__main__":
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

    total_flops = 2*B * H * TLen * D * DV 

    q = torch.randn(B, HQ, TLen, D, dtype=dtype, device=device)
    k = torch.randn(B, HK, TLen, D, dtype=dtype, device=device)
    v = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    g = torch.randn(B, H, TLen, dtype=accum_dtype, device=device)
    do = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)

    from fla.ops.common.chunk_h import chunk_bwd_dh

    dh_ref, _ = chunk_bwd_dh(q, k, v, g, None, None, do, None, None, scale=1.0, chunk_size=BT)
    dh_ref = rearrange(dh_ref, "b h n k v -> b h (n k) v")

    mod = tl.cached(chunk_bwd_kernel_dh, [5,], B, HQ, HK, H,TLen, D, DV, BT, BK, BV, num_stages, num_threads)
    dh = mod(q, k, v, g, do)

    from benchmark.bench_utils import print_debug

    print_debug(dh, dh_ref)

    program = chunk_bwd_kernel_dh( B, HQ, HK, H,TLen, D, DV, BT, BK, BV, num_stages, num_threads)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [5,], tl.TensorSupplyType.Normal)
    
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.2f} ms".format(latency))
    
    from tvm.tl.utils import do_bench
    def run_ref():
        chunk_bwd_dh(q, k, v, g, None, None, do, None, None, scale=1.0, chunk_size=BT)
    latency = do_bench(run_ref, warmup=10, rep=10)
    print("ref: {:.2f} ms".format(latency))








