import torch
from typing import Optional, Tuple

import tvm.tl.language as T
from tvm import tl

from einops import rearrange
import torch.nn.functional as F

def chunk_simple_gla_bwd_dqkg_triton(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    scale: float,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, v.shape[-1]
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device).fill_(-1e9)
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q,
        k,
        v,
        h,
        g,
        do,
        dh,
        dq,
        dk,
        dg,
        scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first
    )
    dg = dg.sum(0)
    dg2 = torch.empty(dg.shape, dtype=torch.float32, device=g.device)
    compute_final_dg[(NT, B*H)](dg, dg2, T=T, BT=BT)
    return dq, dk, dg2

def chunk_bwd_dqkg(
    batch, head, headq, headk, seqlen, dim, dimv,
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
        h: T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore
        g: T.Buffer((batch, head, seqlen), accum_dtype), # type: ignore
        do: T.Buffer((batch, head, seqlen, dimv), dtype), # type: ignore
        dh:  T.Buffer((batch, head, NT*dim, dimv), dtype), # type: ignore

        dq: T.Buffer((batch, headq, seqlen, dim), dtype), # type: ignore
        dk: T.Buffer((batch, headk, seqlen, dim), dtype), # type: ignore
        dg: T.Buffer((NK, batch, head, seqlen), accum_dtype), # type: ignore
    ):
        with T.Kernel(NK, NT, batch*head, threads=thread_num) as (bx, by, bz):
            b_g = T.alloc_fragment((BT), accum_dtype)
            b_g1 = T.alloc_fragment((BT), accum_dtype)
            b_g_last = T.alloc_fragment((1), accum_dtype)
            b_g_T = T.alloc_fragment((BT), accum_dtype)
            b_g_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            b_v_shared = T.alloc_shared((BT, BV), dtype)
            b_do_shared = T.alloc_shared((BT, BV), dtype)
            b_dh_shared = T.alloc_shared((BK, BV),dtype)
            b_dh_local = T.alloc_fragment((BK, BV),dtype)
            b_h_shared = T.alloc_shared((BK, BV),dtype)
            b_h_local = T.alloc_fragment((BK, BV),dtype)
            b_hdh = T.alloc_fragment((1,BK*BV),accum_dtype)
            k_shared = T.alloc_shared((BT,BK), dtype)
            q_shared = T.alloc_shared((BT,BK),dtype)
            q_local = T.alloc_fragment((BT,BK),dtype)
            k_local = T.alloc_fragment((BT,BK),dtype)
            k_local1 = T.alloc_fragment((BT,BK),dtype)

            b_dq = T.alloc_fragment((BT, BK), accum_dtype)
            dq_shared = T.alloc_shared((BT,BK), accum_dtype)
            b_dq1 = T.alloc_fragment((BT, BK), accum_dtype)
            b_dk = T.alloc_fragment((BT, BK), accum_dtype)
            dk_shared = T.alloc_shared((BT,BK), accum_dtype)
            b_dk1 = T.alloc_fragment((BT, BK), accum_dtype)
            b_ds = T.alloc_fragment((BT, BT), accum_dtype)
            b_ds_cast = T.alloc_fragment((BT, BT), dtype)
            b_ds_shared = T.alloc_shared((BT, BT), dtype)
            b_dg_last = T.alloc_fragment((1),accum_dtype)
            b_dg_last_tmp = T.alloc_fragment((1),accum_dtype)
            # b_dg_last_local = T.alloc_local((1),accum_dtype)
            # b_dg_last_shared = T.alloc_shared((1),accum_dtype, scope="shared")
            b_dg_qk = T.alloc_fragment((BT,BK),accum_dtype)
            b_dg = T.alloc_fragment((BT),accum_dtype)

            b_dkk = T.alloc_fragment((1,BT*BK), accum_dtype)
            # b_dkksum = T.alloc_fragment((BT), accum_dtype)
            # b_dkksum_shared = T.alloc_shared((BT), accum_dtype, scope="shared")
            # b_dkksum_T = T.alloc_fragment((1,BT), accum_dtype)

            tx = T.thread_binding(0, thread_num, thread="threadIdx.x")

            bhead = bz % head
            bb = bz // head

            T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_shared)
            T.copy(b_g_shared, b_g)
            b_g_last[0] = b_g_shared[BT-1]

            T.clear(b_dg_last)
            T.clear(b_dg)
            T.clear(b_ds)
            T.clear(b_dq)
            T.clear(b_dk)
            for i_v in T.Pipelined(NV, num_stages=num_stages):
                T.copy(v[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_v_shared)
                T.copy(do[bb,bhead, by*BT:(by+1)*BT, i_v*BV:(i_v+1)*BV], b_do_shared)

                T.copy(dh[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_dh_shared)
                T.copy(h[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, i_v*BV:(i_v+1)*BV], b_h_shared)

                T.gemm(b_do_shared, b_v_shared, b_ds, transpose_A=False, transpose_B=True)
                
                T.copy(b_h_shared, b_h_local)
                T.copy(b_dh_shared, b_dh_local)
                for i,j in T.Parallel(BK,BV):
                    b_hdh[0,i*BV+j] = b_h_local[i,j]* b_dh_local[i,j]
                # tl only support clear=True for sharedmemory reduce
                T.reduce_sum(b_hdh, b_dg_last_tmp,dim=1, clear=True)
                b_dg_last[0] += b_dg_last_tmp[0]
                T.gemm(b_do_shared, b_h_shared, b_dq, transpose_A=False, transpose_B=True)
                T.gemm(b_v_shared, b_dh_shared, b_dk, transpose_A=False, transpose_B=True)

            # T.clear(b_dg_last)
            # for ii in T.serial(NV):
            #     T.copy(dh[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, ii*BV:(ii+1)*BV], b_dh_local)
            #     T.copy(h[bb,bhead,by*dim+bx*BK:by*dim+(bx+1)*BK, ii*BV:(ii+1)*BV], b_h_local)
            #     for i,j in T.Parallel(BK,BV):
            #         b_hdh[0,i*BV+j] = b_h_local[i,j]* b_dh_local[i,j]
            #     T.reduce_sum(b_hdh, b_dg_last,dim=1, clear=False)
                
            T.copy(k[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], k_shared)
            T.copy(q[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK], q_shared)
            
            b_dg_last[0] *= T.exp(b_g_last[0])

            for i,j in T.Parallel(BT,BK):
                b_dq[i,j] *= T.exp(b_g[i])
                b_dq[i,j] *= scale

            for i,j in T.Parallel(BT,BK):
                b_dk[i,j] *= T.exp(-b_g[i]+b_g_last[0])
            
            
            # T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g1)
            # T.copy(g[bb, bhead, by*BT:(by+1)*BT], b_g_T) 
            # faster
            T.copy(b_g_shared, b_g1)
            T.copy(b_g_shared, b_g_T)
            for i,j in T.Parallel(BT,BT):
                b_ds[i,j] = T.if_then_else(
                    i >= j, b_ds[i,j]*scale*T.exp(b_g1[i]-b_g_T[j]), 0
                )
            T.copy(b_ds,b_ds_cast)
            
            for i,j in T.Parallel(BT,BK):
                b_dkk[0,i*BK+j] = b_dk[i,j]
            T.copy(k_shared, k_local1)
            for i,j in T.Parallel(BT,BK):
                b_dkk[0,i*BK+j] *= k_local1[i,j]
            T.reduce_sum(b_dkk, b_dg_last_tmp, dim=1, clear=True)
            b_dg_last[0] += b_dg_last_tmp[0]
            
            T.gemm(b_ds_cast,k_shared,b_dq,transpose_A=False, transpose_B=False)
            T.copy(b_ds_cast, b_ds_shared)
            T.gemm(b_ds_shared, q_shared, b_dk, transpose_A=True,transpose_B=False)
            # implicit cast
            # T.copy(b_dq, dq_shared)
            # T.copy(dq_shared, b_dq1)
            for i,j in T.Parallel(BT,BK):
                b_dg_qk[i,j] = b_dq[i,j]
            T.copy(q_shared, q_local)
            T.copy(k_shared, k_local)
            T.copy(b_dk, dk_shared)
            T.copy(dk_shared, b_dk1)
            for i,j in T.Parallel(BT,BK):
                b_dg_qk[i,j] = b_dg_qk[i,j] * q_local[i,j] - b_dk1[i,j]*k_local[i,j]

            T.reduce_sum(b_dg_qk,b_dg,dim=1,clear=True)
            
            for i in T.Parallel(BT):
                b_dg[i] = T.if_then_else(
                    i < BT-1, b_dg[i], b_dg[i]+b_dg_last[0]
                )
            T.copy(b_dq, dq[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            T.copy(b_dk, dk[bb, bhead, by*BT:(by+1)*BT, bx*BK:(bx+1)*BK])
            T.copy(b_dg, dg[bx, bb, bhead, by*BT:(by+1)*BT])

    return main

if __name__ == "__main__":
    # tl slower than triton: 
    # tl faster: B, H, D, DV = 8, 16, 128, 128, 2048
    # modify tvm.tl reduce_sum 
    from fla.ops.simple_gla.chunk import chunk_simple_gla_bwd_dqkg
    from fla.ops.common.chunk_h import chunk_fwd_h, chunk_bwd_dh

    torch.cuda.manual_seed(0)
    B, H, D, DV = 8, 16, 128, 128 # 64 # H100 fail on 128,64
    TLen = 4096 # 512

    BT, BK, BV = 128, 128, 128
    num_stages = 1
    num_threads = 256
    NT = TLen // BT
    dtype = torch.bfloat16
    accum_dtype = torch.float32
    device = "cuda"
    torch.cuda.manual_seed(0)

    total_flops = 0# 2*B * H * TLen * D * DV 

    q = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    k = torch.randn(B, H, TLen, D, dtype=dtype, device=device)
    v = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    g = F.sigmoid(torch.randn(B, H, TLen, dtype=accum_dtype, device=device)).clamp_min(-5)
    do = torch.randn(B, H, TLen, DV, dtype=dtype, device=device)
    dh = torch.randn(B,H,NT*D,DV, dtype=accum_dtype, device=device)
    dh_ref = rearrange(dh, "b h (n k) v -> b h n k v", k=D).contiguous()
    h = torch.randn(B, H, NT*D, DV, dtype=accum_dtype, device=device)
    h_ref = rearrange(h, "b h (n k) v -> b h n k v", k=D).contiguous()

    h_ref, _ = chunk_fwd_h(
        k,
        v,
        g,
        # scale = k.shape[-1] ** -0.5,
        gk=None,
        gv=None,
        h0 = None,
        output_final_state=False,
        states_in_fp32=False,
        chunk_size = BT
    )
    dh_ref, dh0 = chunk_bwd_dh(
        q,
        k,
        v,
        g,
        gk=None,
        gv=None,
        do=do,
        h0=None,
        dht=None,
        scale=1.0, # k.shape[-1] ** -0.5,
        states_in_fp32=False,
        chunk_size=BT
    )
    # print(dh0.shape) None
    h = rearrange(h_ref, "b h n k v -> b h (n k) v").contiguous()
    dh = rearrange(dh_ref, "b h n k v -> b h (n k) v").contiguous()
    print(h.shape)
    print(dh.shape)

    mod = tl.cached(chunk_bwd_dqkg, [7,8,9], B, H, H, H,TLen, D, DV, BT, BK, BV, num_stages, num_threads)
    dq, dk, dg = mod(
        q,k,v,h.bfloat16(),g,do,dh.bfloat16()
    )
    # print(dq)
    dg = dg.sum(0)
    # TODO: rev cumsum
    dq_ref, dk_ref, dg_ref = chunk_simple_gla_bwd_dqkg(
        do,q,k,v,g,h_ref,dh_ref, scale=1.0, chunk_size=BT
    )
    # print(dq_ref)
    
    from benchmark.bench_utils import print_debug
    print_debug(dq,dq_ref)
    print_debug(dk,dk_ref)
    print_debug(dg,dg_ref)
    
    program = chunk_bwd_dqkg( B, H, H, H,TLen, D, DV, BT, BK, BV, num_stages, num_threads)
    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [7,8,9], tl.TensorSupplyType.Normal)
    
    latency = mod.do_bench(mod.func, n_warmup=10, n_repeat=10, profiler="torch")
    print("{:.2f} ms".format(latency))
    
    from tvm.tl.utils import do_bench
    def run_ref():
        dq_ref, dk_ref, dg_ref = chunk_simple_gla_bwd_dqkg(
            do,q,k,v,g,h_ref,dh_ref, scale=1.0, chunk_size=BT
        )
    latency = do_bench(run_ref, warmup=10, rep=10)
    print("ref: {:.2f} ms".format(latency))
    















